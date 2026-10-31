"""Chains the stages and handles retries. The UI and the CLI both call run()."""
import json
import logging
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image

from . import background, export, image_to_3d, mesh_post, metrics, preprocess, sketch_to_image
from .config import apply_overrides, load_config, resolve_path
from .device import (
    free_gpu_memory,
    hardware_report,
    peak_ram_bytes,
    peak_vram_bytes,
    pick_device,
    reset_peak_vram,
)
from .errors import (
    BackgroundError,
    ConfigError,
    ExportError,
    ImageTo3DError,
    MeshPostError,
    MetricsError,
    PreprocessError,
    Sketch2ShapeError,
    SketchToImageError,
    ValidationError,
    WeightsError,
)

log = logging.getLogger(__name__)
RUN_ID_FORMAT = "%Y%m%d-%H%M%S"
RUN_ID_SUFFIX_LENGTH = 6
STAGES = (
    "preprocess", "sketch_understanding", "sketch_to_image", "background",
    "image_to_3d", "mesh_post", "export", "validation", "metrics",
)
# Another seed cannot fix these, so they stop the run at once
NOT_RETRYABLE = (ConfigError, WeightsError, PreprocessError)
_runs_in_process = 0


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    files: dict = field(default_factory=dict)
    stages: list = field(default_factory=list)
    model: str = ""
    textured: bool = False
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    subject: str = None             # what CLIP read from the sketch, when there was no prompt
    subject_confidence: float = None
    validation: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    attempts: list = field(default_factory=list)
    on_stage: Callable = field(default=None, repr=False)
    on_artifact: Callable = field(default=None, repr=False)
    current_attempt: int = field(default=1, repr=False)


@contextmanager
def _stage(name, error_cls, result, cfg):
    """Time a stage, record its memory peak, and name it in any error."""
    per_gb = cfg["runtime"]["bytes_per_gb"]
    if result.on_stage:
        result.on_stage(name)
    reset_peak_vram()
    start = time.perf_counter()
    try:
        yield
    except Sketch2ShapeError:
        raise
    except Exception as exc:
        hint = "lower resolutions or use remote_gpu" if "out of memory" in str(exc).lower() else None
        raise error_cls(f"{type(exc).__name__}: {exc}", hint=hint) from exc
    record = {
        "stage": name,
        "attempt": result.current_attempt,
        "seconds": round(time.perf_counter() - start, 2),
        "peak_vram_gb": round(peak_vram_bytes() / per_gb, 2),
        "peak_ram_gb": round(peak_ram_bytes() / per_gb, 2),
    }
    result.stages.append(record)
    log.info("%s", record)


def _notify(result, stem, path):
    if result.on_artifact:
        result.on_artifact(stem, path)


def _attempt(number, prepared, guidance, cfg, device, dtype, result, folder, force_3d):
    """One generate-to-export pass. Returns (attempt record, state needed if it is selected)."""
    settings = cfg["sketch_to_image"]
    release = cfg["runtime"]["unload_between_stages"]
    record = {"attempt": number, "seed": cfg["runtime"]["seed"], "controlnet_scale": settings["controlnet_scale"]}
    result.current_attempt = number
    folder.mkdir()

    with _stage("sketch_to_image", SketchToImageError, result, cfg):
        image = sketch_to_image.generate(prepared.control, guidance, cfg, device, dtype)
        if release:
            sketch_to_image.unload()
            free_gpu_memory()
    image.save(folder / "04_generated.png")
    _notify(result, "04_generated", folder / "04_generated.png")

    with _stage("background", BackgroundError, result, cfg):
        rgba, foreground_mask = background.isolate(image, cfg)
        if release:
            background.unload()
    rgba.save(folder / "05_foreground.png")
    _notify(result, "05_foreground", folder / "05_foreground.png")
    gate = metrics.image_gate(prepared.mask, prepared.control, foreground_mask, cfg)
    record.update(
        image_iou=gate["image_iou"], ink_coverage=gate["ink_coverage"],
        clipped_ratio=gate["clipped_ratio"], gate_passed=gate["passed"],
    )
    if not gate["passed"] and not force_3d:
        record["outcome"] = "rejected before 3D"
        return record, None

    with _stage("image_to_3d", ImageTo3DError, result, cfg):
        recon = image_to_3d.reconstruct(rgba, cfg, device)
        if release:
            image_to_3d.unload()
            free_gpu_memory()
    with _stage("mesh_post", MeshPostError, result, cfg):
        mesh = mesh_post.normalize(recon.mesh, cfg)
        record["silhouette_iou"] = round(metrics.silhouette_iou(prepared.mask, mesh, cfg), 4)
    with _stage("export", ExportError, result, cfg):
        exported = export.export_mesh(mesh, folder, cfg)
    with _stage("validation", ValidationError, result, cfg):
        validation = export.validate(exported, recon.textured, cfg)
    record["outcome"] = "valid"
    return record, {"recon": recon, "mesh": mesh, "validation": validation, "exported": exported}


def _rank(record):
    return record["gate_passed"], record["ink_coverage"], record["silhouette_iou"]


def _generate_attempts(prepared, guidance, cfg, device, dtype, result):
    """Try seeds until a result passes every check. Returns (best state, its folder, its record)."""
    retry = cfg["retry"]
    base_seed = cfg["runtime"]["seed"]
    base_scale = cfg["sketch_to_image"]["controlnet_scale"]
    threshold = cfg["metrics"]["min_silhouette_iou"]
    best, best_folder, best_record, last_error = None, None, None, None
    for number in range(1, retry["max_attempts"] + 1):
        attempt_cfg = apply_overrides(cfg, [
            f"runtime.seed={base_seed + (number - 1) * retry['seed_step']}",
            f"sketch_to_image.controlnet_scale={base_scale + (number - 1) * retry['scale_step']}",
        ])
        folder = result.run_dir / f"attempt_{number}"
        is_last_chance = number == retry["max_attempts"] and best is None
        try:
            record, state = _attempt(
                number, prepared, guidance, attempt_cfg, device, dtype, result, folder, is_last_chance
            )
        except NOT_RETRYABLE:
            raise
        except Sketch2ShapeError as exc:
            log.warning("attempt %d failed: %s", number, exc)
            last_error = exc
            result.attempts.append({"attempt": number, "outcome": f"error: {exc}"})
            continue
        result.attempts.append(record)
        if state and (best is None or _rank(record) > _rank(best_record)):
            best, best_folder, best_record = state, folder, record
        if state and record["gate_passed"] and record["silhouette_iou"] >= threshold:
            break
    if best is None:
        raise last_error or ImageTo3DError("no attempt produced a valid model")
    return best, best_folder, best_record


def _publish(best_folder, best, result, cfg):
    """Move the selected attempt's files to the top of the run folder."""
    for item in best_folder.iterdir():
        target = result.run_dir / item.name
        shutil.copytree(item, target) if item.is_dir() else shutil.copy2(item, target)
    result.files["04_generated"] = result.run_dir / "04_generated.png"
    result.files["05_foreground"] = result.run_dir / "05_foreground.png"
    _notify(result, "04_generated", result.files["04_generated"])
    _notify(result, "05_foreground", result.files["05_foreground"])
    for fmt, path in best["exported"].items():
        result.files[fmt] = result.run_dir / path.relative_to(best_folder)
    if not cfg["retry"]["keep_attempts"]:
        for folder in result.run_dir.glob("attempt_*"):
            shutil.rmtree(folder)


def _write_meta(result, base, cfg):
    meta = {
        **base,
        "run_id": result.run_id,
        "subject": result.subject,
        "subject_confidence": result.subject_confidence,
        "model": result.model,
        "textured": result.textured,
        "notes": result.notes,
        "warnings": result.warnings,
        "attempts": result.attempts,
        "stages": result.stages,
        "total_seconds": round(sum(stage["seconds"] for stage in result.stages), 2),
        "peak_vram_gb": max((stage["peak_vram_gb"] for stage in result.stages), default=0),
        "peak_ram_gb": max((stage["peak_ram_gb"] for stage in result.stages), default=0),
        "metrics": result.metrics,
        "validation": result.validation,
        "files": {name: str(path) for name, path in result.files.items()},
        "hardware": hardware_report(cfg),
        "config": cfg,
    }
    with open(result.run_dir / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def _save_image(image, name, result):
    path = result.run_dir / name
    image.save(path)
    result.files[path.stem] = path
    _notify(result, path.stem, path)


def run(sketch, prompt=None, overrides=None, cfg=None, on_stage=None, on_artifact=None):
    """on_stage is called with each stage name from STAGES as it starts; on_artifact with (name, path) per image."""
    global _runs_in_process
    cfg = apply_overrides(cfg or load_config(), overrides)
    run_id = f"{datetime.now().strftime(RUN_ID_FORMAT)}-{uuid.uuid4().hex[:RUN_ID_SUFFIX_LENGTH]}"
    result = RunResult(
        run_id, resolve_path(cfg["paths"]["outputs_dir"]) / run_id, on_stage=on_stage, on_artifact=on_artifact
    )
    result.run_dir.mkdir(parents=True)
    device, dtype = pick_device(cfg)
    base = {
        "prompt": prompt, "device": device, "dtype": dtype, "seed": cfg["runtime"]["seed"],
        "cold_start": _runs_in_process == 0,
    }
    _runs_in_process += 1
    try:
        with _stage("preprocess", PreprocessError, result, cfg):
            prepared = preprocess.prepare(sketch, cfg)
        _save_image(prepared.clean, "01_clean_sketch.png", result)
        _save_image(prepared.control, "02_control_map.png", result)
        _save_image(prepared.mask, "03_sketch_mask.png", result)

        guidance = prompt
        if not (prompt or "").strip() and cfg["sketch_to_image"]["auto_subject"]:
            with _stage("sketch_understanding", SketchToImageError, result, cfg):
                guidance, result.subject_confidence = sketch_to_image.infer_subject(prepared.clean, cfg, device)
                free_gpu_memory()
            result.subject = guidance

        best, best_folder, selected = _generate_attempts(prepared, guidance, cfg, device, dtype, result)
        recon = best["recon"]
        result.model, result.textured, result.notes = recon.model, recon.textured, recon.notes
        result.validation = best["validation"]
        _publish(best_folder, best, result, cfg)

        result.current_attempt = selected["attempt"]
        with _stage("metrics", MetricsError, result, cfg):
            views = metrics.render_views(best["mesh"], cfg["metrics"])
            strip = Image.new("RGB", (sum(view.width for view in views), views[0].height))
            for index, view in enumerate(views):
                strip.paste(view, (index * view.width, 0))
            _save_image(strip, "06_views.png", result)
            match_view, mesh_ink_coverage = metrics.sketch_match(
                prepared.control, best["mesh"], views[0], cfg["metrics"]
            )
            _save_image(match_view, "07_sketch_match.png", result)
            text = sketch_to_image.build_prompt(guidance, cfg) if guidance else None
            result.metrics = {
                "selected_attempt": selected["attempt"],
                "seed": selected["seed"],
                "mesh_ink_coverage": mesh_ink_coverage,
                "silhouette_iou": selected["silhouette_iou"],
                "image_iou": selected["image_iou"],
                "ink_coverage": selected["ink_coverage"],
                "clipped_ratio": selected["clipped_ratio"],
                **metrics.clip_scores(prepared.clean, views, text, cfg, device),
                **metrics.geometry_health(best["mesh"], recon.clean_stats),
            }
            free_gpu_memory()
        limits = cfg["metrics"]
        if selected["ink_coverage"] < limits["min_ink_coverage"]:
            result.warnings.append(
                f"part of the sketch may be missing: only {selected['ink_coverage']:.0%} of its strokes landed on the object"
            )
        if selected["silhouette_iou"] < limits["min_silhouette_iou"]:
            result.warnings.append(
                f"low sketch fidelity: silhouette IoU {selected['silhouette_iou']} is below {limits['min_silhouette_iou']}"
            )
        if selected["clipped_ratio"] > limits["max_clipped_ratio"]:
            result.warnings.append("the object was cut off by the image frame, so part of it may be missing")
    except Sketch2ShapeError as exc:
        _write_meta(result, {**base, "status": "failed", "error": str(exc)}, cfg)
        raise
    _write_meta(result, {**base, "status": "ok"}, cfg)
    return result


def _variant_rank(result):
    return not result.warnings, result.metrics["mesh_ink_coverage"], result.metrics["silhouette_iou"]


def run_variants(sketch, prompt=None, overrides=None, cfg=None, on_stage=None, on_artifact=None):
    """Generate several models from one sketch. Returns (results best first, errors of failed variants).

    on_stage, if given, is called with (variant number, stage name); on_artifact with (variant number, name, path).
    """
    cfg = apply_overrides(cfg or load_config(), overrides)
    settings = cfg["variants"]
    base_seed = cfg["runtime"]["seed"]
    results, errors = [], []
    for number in range(1, settings["count"] + 1):
        variant_overrides = [
            f"runtime.seed={base_seed + (number - 1) * settings['seed_step']}",
            f"retry.max_attempts={settings['max_attempts']}",
        ]
        report = (lambda name, n=number: on_stage(n, name)) if on_stage else None
        artifact = (lambda stem, path, n=number: on_artifact(n, stem, path)) if on_artifact else None
        try:
            results.append(run(sketch, prompt, variant_overrides, cfg, report, artifact))
        except NOT_RETRYABLE:
            raise
        except Sketch2ShapeError as exc:
            log.warning("variant %d failed: %s", number, exc)
            errors.append(exc)
    if not results:
        raise errors[-1]
    results.sort(key=_variant_rank, reverse=True)
    return results, [str(error) for error in errors]
