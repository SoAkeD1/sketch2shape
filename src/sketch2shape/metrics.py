"""Measured quality of a run: sketch fidelity, semantic match, view consistency, geometry health.

Rendering is plain CPU rasterization with OpenCV, so metrics work on any machine and GPU.
"""
import itertools
import math

import cv2
import numpy as np
import torch
from PIL import Image

from .device import torch_device
from .sketch_to_image import weights_path

OPAQUE = 255


def _face_colors(mesh):
    visual = mesh.visual
    if visual.kind == "texture":
        material = visual.material
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        texture = np.asarray(image.convert("RGB"))
        height, width = texture.shape[:2]
        uv = np.asarray(visual.uv)[mesh.faces].mean(axis=1) % 1.0
        rows = ((1 - uv[:, 1]) * (height - 1)).astype(int)
        return texture[rows, (uv[:, 0] * (width - 1)).astype(int)].astype(np.float32)
    return np.asarray(visual.vertex_colors)[mesh.faces][:, :, :3].mean(axis=1).astype(np.float32)


def _project(vertices, azimuth_deg, settings):
    """Turn the mesh about Y, then apply a perspective camera on the +Z axis."""
    angle = math.radians(azimuth_deg)
    cos, sin = math.cos(angle), math.sin(angle)
    x = vertices[:, 0] * cos + vertices[:, 2] * sin
    z = -vertices[:, 0] * sin + vertices[:, 2] * cos
    size = settings["render_size"]
    focal = (size / 2) / math.tan(math.radians(settings["camera_fov_deg"]) / 2)
    depth = settings["camera_distance"] - z
    pixels = np.stack([size / 2 + focal * x / depth, size / 2 - focal * vertices[:, 1] / depth], axis=1)
    return np.rint(pixels).astype(np.int32), z


def silhouette(mesh, settings, azimuth_deg=0.0):
    size = settings["render_size"]
    canvas = np.zeros((size, size), np.uint8)
    pixels, _ = _project(np.asarray(mesh.vertices), azimuth_deg, settings)
    for triangle in pixels[mesh.faces]:
        cv2.fillConvexPoly(canvas, triangle, OPAQUE)
    return canvas


def render_view(mesh, colors, settings, azimuth_deg):
    """Painter's algorithm with simple headlight shading on a white background."""
    size = settings["render_size"]
    canvas = np.full((size, size, 3), OPAQUE, np.uint8)
    pixels, z = _project(np.asarray(mesh.vertices), azimuth_deg, settings)
    faces = np.asarray(mesh.faces)
    angle = math.radians(azimuth_deg)
    view = np.array([-math.sin(angle), 0.0, math.cos(angle)])
    ambient = settings["render_ambient"]
    shade = ambient + (1 - ambient) * np.abs(np.asarray(mesh.face_normals) @ view)
    shaded = colors * shade[:, None]
    for index in np.argsort(z[faces].mean(axis=1)):
        cv2.fillConvexPoly(canvas, pixels[faces[index]], shaded[index].tolist())
    return Image.fromarray(canvas)


def render_views(mesh, settings):
    colors = _face_colors(mesh)
    count = settings["n_views"]
    return [render_view(mesh, colors, settings, 360.0 * i / count) for i in range(count)]


def _fit_to(image, mask, size, fit, background=0):
    """Crop image to the mask's shape and center it, so comparisons ignore framing."""
    canvas = np.full((size, size) + image.shape[2:], background, np.uint8)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return canvas
    crop = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = fit * size / max(crop.shape[:2])
    width = max(1, round(crop.shape[1] * scale))
    height = max(1, round(crop.shape[0] * scale))
    crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    y, x = (size - height) // 2, (size - width) // 2
    canvas[y:y + height, x:x + width] = crop
    return canvas


def _fit(mask, settings):
    return _fit_to(mask, mask, settings["iou_size"], settings["iou_fit"])


def sketch_match(sketch_ink, mesh, front_view, settings):
    """Draw the user's strokes over the mesh's front view. Returns (image, share of strokes on the mesh).

    Strokes on the model use the match color and strokes that missed use the miss color.
    """
    size, fit = settings["match_view_size"], settings["iou_fit"]
    shape = silhouette(mesh, settings)
    ink = np.asarray(sketch_ink.convert("L"))
    fitted_ink = _fit_to(ink, ink, size, fit) > 0
    tolerance = settings["ink_tolerance_px"]
    fitted_shape = _fit_to(shape, shape, size, fit)
    grown = cv2.dilate(fitted_shape, np.ones((tolerance, tolerance), np.uint8)) > 0
    canvas = _fit_to(np.asarray(front_view), shape, size, fit, background=OPAQUE)
    canvas[fitted_ink & grown] = settings["match_color"]
    rows, columns = np.indices(fitted_ink.shape)
    period = settings["miss_dash_px"]
    # Missed strokes are dashed too, so the two states don't rely on color alone
    canvas[fitted_ink & ~grown & (((rows + columns) // period) % 2 == 0)] = settings["miss_color"]
    coverage = float((fitted_ink & grown).sum() / max(1, fitted_ink.sum()))
    return Image.fromarray(canvas), round(coverage, 4)


def _iou(first, second):
    first, second = first > 0, second > 0
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 0.0


def silhouette_iou(sketch_mask, mesh, cfg):
    settings = cfg["metrics"]
    return _iou(_fit(np.asarray(sketch_mask), settings), _fit(silhouette(mesh, settings), settings))


def image_gate(sketch_mask, sketch_ink, foreground_mask, cfg):
    """Cheap check before the slow 3D stage. All masks share the generated image's frame.

    Ink coverage is the share of the user's strokes that landed on the object. Unlike IoU it
    still works on broken lines, and it drops when a part of the drawing goes missing.
    """
    settings = cfg["metrics"]
    foreground = foreground_mask.astype(np.uint8) * OPAQUE
    height, width = foreground.shape
    size = (width, height)
    sketch = cv2.resize(np.asarray(sketch_mask), size, interpolation=cv2.INTER_NEAREST)
    ink = cv2.resize(np.asarray(sketch_ink.convert("L")), size, interpolation=cv2.INTER_NEAREST) > 0
    tolerance = settings["ink_tolerance_px"]
    grown = cv2.dilate(foreground, np.ones((tolerance, tolerance), np.uint8)) > 0
    border = np.concatenate([foreground[0], foreground[-1], foreground[:, 0], foreground[:, -1]])
    report = {
        "image_iou": round(_iou(sketch, foreground), 4),
        "ink_coverage": round(float((ink & grown).sum() / max(1, ink.sum())), 4),
        "clipped_ratio": round(float((border > 0).mean()), 4),
    }
    report["passed"] = (
        report["ink_coverage"] >= settings["min_ink_coverage"]
        and report["clipped_ratio"] <= settings["max_clipped_ratio"]
    )
    return report


def _cosine_matrix(first, second):
    first = first / first.norm(dim=-1, keepdim=True)
    second = second / second.norm(dim=-1, keepdim=True)
    return first @ second.T


def clip_scores(clean_sketch, views, text, cfg, device):
    """Cosine similarities in CLIP space. Weights are loaded for this call only."""
    from transformers import CLIPModel, CLIPProcessor

    path = weights_path(cfg["metrics"]["clip_weights"], cfg)
    target = torch_device(device)
    model = CLIPModel.from_pretrained(path).to(target)
    processor = CLIPProcessor.from_pretrained(path)
    images = processor(images=[clean_sketch] + views, return_tensors="pt").to(target)
    with torch.no_grad():
        features = model.visual_projection(model.vision_model(**images).pooler_output)
        sketch, rendered = features[:1], features[1:]
        pairs = list(itertools.combinations(range(len(views)), 2))
        similarity = _cosine_matrix(rendered, rendered)
        scores = {
            "clip_sketch_to_views": round(float(_cosine_matrix(sketch, rendered).mean()), 4),
            "clip_view_consistency": round(float(sum(similarity[a, b] for a, b in pairs) / len(pairs)), 4),
        }
        if text:
            tokens = processor(text=[text], return_tensors="pt", padding=True).to(target)
            encoded = model.text_projection(model.text_model(**tokens).pooler_output)
            scores["clip_text_to_views"] = round(float(_cosine_matrix(encoded, rendered).mean()), 4)
    return scores


def geometry_health(mesh, clean_stats):
    extents = mesh.extents
    faces_in = clean_stats.get("faces_in") or 1
    return {
        "faces": len(mesh.faces),
        "watertight": clean_stats.get("watertight"),
        "degenerate_face_ratio": round(clean_stats.get("degenerate_removed", 0) / faces_in, 6),
        "floaters_removed": clean_stats.get("floaters_removed"),
        "bbox_aspect_ratio": round(float(extents.min() / extents.max()), 4),
    }
