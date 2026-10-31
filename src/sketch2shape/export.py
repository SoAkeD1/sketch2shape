"""Stage 7. Write .glb / .obj, then reload every file and check it. A failed check raises."""
from pathlib import Path

import numpy as np
import trimesh

from .errors import ExportError, ValidationError

MODEL_STEM = "model"
OBJ_SUBDIR = "obj"
BYTES_PER_MB = 1024 * 1024


def export_mesh(mesh, run_dir, cfg):
    """Returns {format: path}. Each .obj goes in its own folder with its .mtl and texture."""
    paths = {}
    for fmt in cfg["export"]["formats"]:
        folder = Path(run_dir) / OBJ_SUBDIR if fmt == "obj" else Path(run_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{MODEL_STEM}.{fmt}"
        try:
            if fmt == "glb":
                mesh.export(path, include_normals=True)
            elif fmt == "obj":
                mesh.export(path, include_normals=True, include_texture=True)
            else:
                raise ExportError(f"unsupported format {fmt!r}", hint="use glb or obj")
        except ExportError:
            raise
        except Exception as exc:
            raise ExportError(f"could not write {path.name}: {exc}") from exc
        paths[fmt] = path
    if not paths:
        raise ExportError("export.formats is empty", hint="set it to glb and/or obj")
    return paths


def _has_texture(mesh):
    visual = mesh.visual
    if visual.kind != "texture" or visual.uv is None or len(visual.uv) == 0:
        return False
    material = visual.material
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    return image is not None


def _has_vertex_colors(mesh):
    return mesh.visual.kind == "vertex"


def check_file(path, expect_texture, cfg):
    """Reload a file and run every check. Returns the report; raises nothing itself."""
    settings = cfg["export"]
    path = Path(path)
    report = {"file": path.name, "size_mb": round(path.stat().st_size / BYTES_PER_MB, 2)}
    failed = []
    if report["size_mb"] > settings["max_file_mb"]:
        failed.append(f"file is {report['size_mb']} MB, limit is {settings['max_file_mb']} MB")
    try:
        mesh = trimesh.load(path, force="mesh")
    except Exception as exc:
        report.update(failed=[f"file does not open: {exc}"], passed=False)
        return report

    report["vertices"], report["faces"] = len(mesh.vertices), len(mesh.faces)
    if report["vertices"] < settings["min_vertices"]:
        failed.append(f"only {report['vertices']} vertices")
    if report["faces"] < settings["min_faces"]:
        failed.append(f"only {report['faces']} faces")
    if report["vertices"] and not np.isfinite(mesh.vertices).all():
        failed.append("vertices contain NaN or infinity")
    elif report["vertices"]:
        extents = mesh.extents
        report["extents"] = [round(float(e), 4) for e in extents]
        ratio = float(extents.min() / extents.max()) if extents.max() > 0 else 0.0
        report["min_axis_ratio"] = round(ratio, 4)
        if ratio < settings["min_axis_ratio"]:
            failed.append(f"mesh is flat: thinnest side is {ratio:.1%} of the largest")
    report["textured"] = _has_texture(mesh)
    report["vertex_colors"] = _has_vertex_colors(mesh)
    if expect_texture and not report["textured"]:
        failed.append("texture was expected but did not survive reload")
    if not expect_texture and not report["vertex_colors"]:
        failed.append("vertex colors were expected but did not survive reload")
    report.update(failed=failed, passed=not failed)
    return report


def validate(paths, expect_texture, cfg):
    """Check all exported files. Raises ValidationError naming each failed check."""
    reports = {fmt: check_file(path, expect_texture, cfg) for fmt, path in paths.items()}
    problems = [f"{fmt}: {problem}" for fmt, report in reports.items() for problem in report["failed"]]
    if problems:
        raise ValidationError("; ".join(problems))
    return reports
