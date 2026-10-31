"""End-to-end regression check. Run: python tests/regression_check.py

Exercises every promised feature on the real models, with reduced settings for speed.
Exits 1 if any check fails. Needs the weights from scripts/download_weights.py.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402

from sketch2shape import export, image_to_3d, mesh_post, pipeline, preprocess  # noqa: E402
from sketch2shape.config import load_config  # noqa: E402
from sketch2shape.device import hardware_report  # noqa: E402
from sketch2shape.errors import ConfigError, PreprocessError, Sketch2ShapeError, ValidationError  # noqa: E402

SKETCH = ROOT / "examples" / "vase" / "sketch.png"
FAST = [
    "sketch_to_image.steps=8",
    "image_to_3d.triposr.mc_resolution=96",
    "image_to_3d.triposr.texture_resolution=256",
    "retry.max_attempts=2",
    "metrics.min_ink_coverage=0.0",
    "metrics.min_silhouette_iou=0.0",
    "metrics.max_clipped_ratio=1.0",
]
EXPECTED_FILES = [
    "01_clean_sketch.png", "02_control_map.png", "03_sketch_mask.png", "04_generated.png", "05_foreground.png",
    "06_views.png", "07_sketch_match.png", "model.glb", "obj/model.obj", "obj/material.mtl", "run_meta.json",
]
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((name, True, detail or ""))
        print(f"PASS  {name}  {detail or ''}", flush=True)
    except Exception as exc:  # A test harness reports every failure and keeps going
        results.append((name, False, repr(exc)[:300]))
        print(f"FAIL  {name}  {repr(exc)[:300]}", flush=True)


def rejects(array, error=PreprocessError):
    try:
        preprocess.prepare(array, CFG)
    except error as exc:
        return str(exc)[:60]
    raise AssertionError("was accepted")


CFG = load_config()
BLANK = np.full((512, 512), 255, np.uint8)
ink = np.asarray(Image.open(SKETCH).convert("L"))


def config_precedence():
    os.environ["S2S_SKETCH_TO_IMAGE__STEPS"] = "99"
    try:
        assert load_config()["sketch_to_image"]["steps"] == 99
        assert load_config(overrides=["sketch_to_image.steps=7"])["sketch_to_image"]["steps"] == 7
    finally:
        del os.environ["S2S_SKETCH_TO_IMAGE__STEPS"]
    try:
        load_config(overrides=["nope.x=1"])
    except ConfigError:
        return "yaml < env < cli, unknown key rejected"
    raise AssertionError("unknown key accepted")


def preprocess_accepts():
    transparent = np.dstack([np.zeros_like(ink)] * 3 + [255 - ink])
    inverted = 255 - ink
    small = np.full((768, 768), 255, np.uint8)
    small[20:276, 480:736] = np.asarray(Image.fromarray(ink).resize((256, 256)))
    for array in (ink, transparent, inverted, small):
        prepared = preprocess.prepare(array, CFG)
        assert prepared.clean.size == prepared.control.size == prepared.mask.size
    prepared = preprocess.prepare(str(SKETCH), CFG)
    assert np.asarray(prepared.mask).any()
    return "array, path, transparent, inverted, off-center"


def assert_valid_run(result, textured=True):
    for name in EXPECTED_FILES:
        if name.endswith(".mtl") and not textured:
            continue  # Vertex-color models have no material file
        assert (result.run_dir / name).is_file(), f"missing {name}"
    meta = json.load(open(result.run_dir / "run_meta.json", encoding="utf-8"))
    assert meta["status"] == "ok"
    for key in ("config", "hardware", "stages", "attempts", "metrics", "validation", "total_seconds"):
        assert key in meta, f"meta lacks {key}"
    for key in ("mesh_ink_coverage", "ink_coverage", "silhouette_iou", "clip_sketch_to_views",
                "clip_view_consistency", "watertight", "faces", "bbox_aspect_ratio", "seed"):
        assert key in meta["metrics"], f"metrics lacks {key}"
    for fmt in ("glb", "obj"):
        assert meta["validation"][fmt]["passed"]
        assert meta["validation"][fmt]["textured"] is textured
    mesh = trimesh.load(result.run_dir / "model.glb", force="mesh")
    assert len(mesh.faces) > 0 and mesh.extents.min() / mesh.extents.max() > 0.05
    assert abs(float(mesh.extents.max()) - 1.0) < 1e-3, "not normalized"
    return meta


def run_no_prompt():
    seen = []
    result = pipeline.run(ink, overrides=FAST, on_stage=seen.append)
    meta = assert_valid_run(result)
    assert result.subject, "CLIP did not name the sketch"
    assert set(seen) <= set(pipeline.STAGES) and "sketch_understanding" in seen and "metrics" in seen
    return f"subject={result.subject} faces={meta['metrics']['faces']} stages={len(seen)}"


def run_prompt():
    result = pipeline.run(ink, "a blue glass vase", overrides=FAST)
    assert_valid_run(result)
    assert result.subject is None
    return "prompt skips CLIP naming"


def run_path_input():
    assert_valid_run(pipeline.run(str(SKETCH), overrides=FAST))
    return "upload path"


def run_vertex_colors():
    result = pipeline.run(ink, overrides=FAST + ["image_to_3d.triposr.bake_texture=false"])
    assert_valid_run(result, textured=False)
    return "vertex colors survive"


def seed_reproducible():
    a = pipeline.run(ink, overrides=FAST + ["runtime.seed=7"])
    b = pipeline.run(ink, overrides=FAST + ["runtime.seed=7"])
    assert np.array_equal(np.asarray(Image.open(a.files["04_generated"])), np.asarray(Image.open(b.files["04_generated"])))
    assert a.metrics["faces"] == b.metrics["faces"] and a.metrics["seed"] == 7
    return "same image, same faces"


def decimate():
    result = pipeline.run(ink, overrides=FAST + ["mesh_post.target_faces=5000"])
    assert result.metrics["faces"] == 5000
    return "exactly 5000 faces"


def retry_recovers():
    real, calls = export.validate, {"n": 0}

    def flaky(paths, expect, cfg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValidationError("forced")
        return real(paths, expect, cfg)

    export.validate = flaky
    try:
        result = pipeline.run(ink, overrides=FAST)
    finally:
        export.validate = real
    assert result.attempts[0]["outcome"].startswith("error") and result.attempts[1]["outcome"] == "valid"
    return "second attempt used"


def never_false_success():
    def always(paths, expect, cfg):
        raise ValidationError("forced")

    real, export.validate = export.validate, always
    try:
        pipeline.run(ink, overrides=FAST)
    except ValidationError as exc:
        run = sorted((ROOT / "outputs").glob("*/run_meta.json"))[-1]
        assert json.load(open(run, encoding="utf-8"))["status"] == "failed"
        return str(exc)[:40]
    finally:
        export.validate = real
    raise AssertionError("reported success")


def fallback_used():
    class Broken:
        name = "broken"

        def __init__(self, cfg, device):
            pass

        def reconstruct(self, rgba, cfg):
            raise RuntimeError("forced")

    image_to_3d.BACKENDS["broken"] = Broken
    try:
        result = pipeline.run(ink, overrides=FAST + ["image_to_3d.primary=broken", "image_to_3d.fallback=triposr"])
        assert result.model == "triposr" and any("fallback" in note for note in result.notes)
    finally:
        del image_to_3d.BACKENDS["broken"]
    return "fallback model took over"


def variants_ranked():
    results_, errors = pipeline.run_variants(ink, overrides=FAST + ["variants.count=2"])
    assert not errors and len(results_) == 2
    seeds = {r.metrics["seed"] for r in results_}
    assert len(seeds) == 2
    assert pipeline._variant_rank(results_[0]) >= pipeline._variant_rank(results_[1])
    assert_valid_run(results_[0])
    return f"seeds={sorted(seeds)}"


def validation_rejects():
    folder = Path(tempfile.mkdtemp())
    flat = trimesh.creation.box((1, 1, 1e-4)).subdivide().subdivide().subdivide()
    flat.visual = trimesh.visual.ColorVisuals(flat, vertex_colors=np.tile([200, 0, 0, 255], (len(flat.vertices), 1)))
    flat.export(folder / "flat.glb")
    plain = trimesh.creation.icosphere(subdivisions=4)
    plain.export(folder / "plain.glb")
    (folder / "broken.glb").write_bytes(b"not a model")
    for path, textured in (("flat.glb", False), ("plain.glb", True), ("broken.glb", False)):
        try:
            export.validate({"glb": folder / path}, textured, CFG)
        except ValidationError:
            continue
        raise AssertionError(f"{path} accepted")
    try:
        mesh_post.clean_geometry(trimesh.Trimesh(), CFG)
    except Sketch2ShapeError:
        return "flat, lost texture, corrupt, empty"
    raise AssertionError("empty mesh accepted")


def cli_run(*args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    command = [sys.executable, "-W", "ignore", str(ROOT / "cli.py"), *args]
    return subprocess.run(command, capture_output=True, text=True, env=env, cwd=ROOT)


def cli_good():
    args = ["--input", str(SKETCH)]
    for item in FAST:
        args += ["--set", item]
    done = cli_run(*args)
    assert done.returncode == 0, done.stderr[-200:]
    assert Path(json.loads(done.stdout)["files"]["glb"]).is_file()
    return "exit 0, .glb exists"


def cli_bad():
    assert cli_run("--input", "nope.png").returncode == 1
    assert cli_run("--input", "nope.png", "--set", "nope.x=1").returncode == 1
    return "exit 1 on bad file and bad key"


def app_builds():
    import app as app_module

    blocks = app_module.build_app(CFG)
    kinds = [type(block).__name__ for block in blocks.blocks.values()]
    for needed in ("Sketchpad", "Image", "Textbox", "Number", "Slider", "Button", "Model3D", "Gallery", "File", "JSON"):
        assert needed in kinds, f"app lost {needed}"
    labels = {getattr(block, "value", None) for block in blocks.blocks.values() if type(block).__name__ == "Button"}
    assert "Generate" in labels or any(str(label).startswith("Generate") for label in labels)
    assert any(str(label).startswith("Generate 3") or "variants" in str(label) for label in labels)
    return "all input and output controls present"


check("config: precedence and unknown keys", config_precedence)
check("device: hardware report", lambda: str(hardware_report(CFG)["selected_device"]))
check("preprocess: accepts every input kind", preprocess_accepts)
check("preprocess: rejects blank", lambda: rejects(BLANK))
check("preprocess: rejects solid black", lambda: rejects(np.zeros((512, 512), np.uint8)))
check("preprocess: rejects noise", lambda: rejects(np.random.default_rng(1).integers(0, 255, (512, 512), np.uint8)))
check("preprocess: rejects missing file", lambda: rejects("does_not_exist.png"))
check("pipeline: no prompt", run_no_prompt)
check("pipeline: with prompt", run_prompt)
check("pipeline: upload path", run_path_input)
check("pipeline: vertex colors", run_vertex_colors)
check("pipeline: decimation", decimate)
check("pipeline: same seed reproduces", seed_reproducible)
check("retry: recovers from a failure", retry_recovers)
check("retry: never reports false success", never_false_success)
check("fallback: second model takes over", fallback_used)
check("variants: ranked, distinct seeds", variants_ranked)
check("validation: rejects bad meshes", validation_rejects)
check("cli: good input", cli_good)
check("cli: bad input", cli_bad)
check("app: all controls present", app_builds)

failed = [name for name, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)} of {len(results)} passed")
for name in failed:
    print("FAILED:", name)
sys.exit(1 if failed else 0)
