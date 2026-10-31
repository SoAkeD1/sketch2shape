"""Stage 2. Sketch (canvas array, upload or path) -> clean sketch, control map, filled mask."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .errors import PreprocessError

OPAQUE = 255


@dataclass
class PreparedSketch:
    clean: Image.Image    # dark lines on white
    control: Image.Image  # white lines on black, as the scribble ControlNet expects
    mask: Image.Image     # filled silhouette of the sketch, used by metrics


def _load(sketch):
    if isinstance(sketch, (str, Path)):
        try:
            sketch = Image.open(sketch)
            sketch.load()
        except OSError as exc:
            raise PreprocessError(f"could not read image file: {exc}") from exc
    elif isinstance(sketch, np.ndarray):
        sketch = Image.fromarray(sketch)
    if not isinstance(sketch, Image.Image):
        raise PreprocessError(f"unsupported sketch type: {type(sketch).__name__}")
    return ImageOps.exif_transpose(sketch)


def _to_gray(image, cfg):
    """Return dark-on-light gray. On a transparent canvas, ink is whatever is opaque."""
    work = image.convert("RGBA")
    work.thumbnail((cfg["work_size"], cfg["work_size"]))
    rgba = np.asarray(work)
    alpha = rgba[..., 3]
    if alpha.min() < OPAQUE and (alpha == 0).any():
        return OPAQUE - alpha
    gray = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    return OPAQUE - gray if np.median(border) < cfg["dark_background_level"] else gray


def _ink_mask(gray, cfg):
    denoised = cv2.fastNlMeansDenoising(gray, None, h=cfg["denoise_strength"])
    grid = cfg["contrast_grid"]
    clahe = cv2.createCLAHE(clipLimit=cfg["contrast_clip_limit"], tileGridSize=(grid, grid))
    contrasted = clahe.apply(denoised)
    local = cv2.adaptiveThreshold(
        contrasted, OPAQUE, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        cfg["binarize_block_size"], cfg["binarize_c"],
    )
    ink = np.where(denoised < cfg["ink_dark_level"], OPAQUE, local).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= cfg["min_component_area"])
    keep = keep[keep != 0]
    return (np.isin(labels, keep) * OPAQUE).astype(np.uint8)


def _center_square(ink, cfg):
    """Crop to the ink and center it, so an off-center sketch is handled like a centered one."""
    ys, xs = np.nonzero(ink)
    top, left, bottom, right = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
    side = int(max(bottom - top, right - left) / (1 - 2 * cfg["pad_ratio"]))
    canvas = np.zeros((side, side), np.uint8)
    y = (side - (bottom - top)) // 2
    x = (side - (right - left)) // 2
    canvas[y:y + bottom - top, x:x + right - left] = ink[top:bottom, left:right]
    resized = cv2.resize(canvas, (cfg["size"], cfg["size"]), interpolation=cv2.INTER_AREA)
    return (resized > cfg["resize_ink_level"]).astype(np.uint8) * OPAQUE


def _filled_mask(ink, cfg):
    size = cfg["fill_close_kernel"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
    # Flood the outside from a border margin; whatever it cannot reach is the object
    padded = cv2.copyMakeBorder(closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), OPAQUE)
    outside = flood_mask[2:-2, 2:-2]
    return ((outside == 0) * OPAQUE).astype(np.uint8)


def prepare(sketch, cfg):
    settings = cfg["preprocess"]
    ink = _ink_mask(_to_gray(_load(sketch), settings), settings)
    ratio = np.count_nonzero(ink) / ink.size
    if ratio < settings["min_ink_ratio"]:
        raise PreprocessError("the sketch looks empty", hint="draw something or upload a clearer image")
    if ratio > settings["max_ink_ratio"]:
        raise PreprocessError(
            f"the image is {ratio:.0%} ink, which does not look like a line sketch",
            hint="use line art on a plain background",
        )
    marks = cv2.connectedComponents(ink)[0] - 1
    if marks > settings["max_components"]:
        raise PreprocessError(
            f"the image has {marks} separate marks, which does not look like a line sketch",
            hint="use line art on a plain background",
        )
    centered = _center_square(ink, settings)
    width = settings["control_line_width"]
    control = cv2.dilate(centered, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width)))
    return PreparedSketch(
        clean=Image.fromarray(OPAQUE - centered).convert("RGB"),
        control=Image.fromarray(control).convert("RGB"),
        mask=Image.fromarray(_filled_mask(centered, settings)),
    )
