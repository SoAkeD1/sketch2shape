"""Plain Gradio front end. All work happens in pipeline.py."""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr  # noqa: E402

from sketch2shape import pipeline  # noqa: E402
from sketch2shape.config import load_config, resolve_path  # noqa: E402
from sketch2shape.errors import Sketch2ShapeError  # noqa: E402

DRAW, UPLOAD = "draw", "upload"
INTERMEDIATES = {
    "01_clean_sketch": "Clean sketch",
    "02_control_map": "Control map",
    "03_sketch_mask": "Sketch mask",
    "04_generated": "Generated image",
    "05_foreground": "Foreground",
    "06_views": "Mesh from all sides",
}
DETAIL_KEYS = (
    "run_id", "prompt", "subject", "subject_confidence", "device", "model", "textured", "warnings",
    "metrics", "attempts", "notes", "total_seconds", "peak_vram_gb", "peak_ram_gb", "stages", "validation",
)


def _pick_sketch(mode, drawn, uploaded):
    if mode == UPLOAD:
        if uploaded is None:
            raise gr.Error("Upload an image first, or switch to the Draw tab.")
        return uploaded
    composite = drawn.get("composite") if isinstance(drawn, dict) else None
    if composite is None:
        raise gr.Error("Draw something first, or switch to the Upload tab.")
    return composite


def _overrides(seed, steps, scale):
    return [
        f"runtime.seed={int(seed)}",
        f"sketch_to_image.steps={int(steps)}",
        f"sketch_to_image.controlnet_scale={float(scale)}",
    ]


def _read_meta(run_dir):
    with open(Path(run_dir) / "run_meta.json", encoding="utf-8") as handle:
        return json.load(handle)


def _badge(meta):
    scores = meta["metrics"]
    text = (
        f"### Sketch match: {scores['mesh_ink_coverage']:.0%} of your strokes are on the 3D model\n"
        f"Silhouette IoU {scores['silhouette_iou']:.2f} · seed {scores['seed']} · {meta['total_seconds']:.0f} s"
    )
    for warning in meta["warnings"]:
        text += f"\n\n**Warning:** {warning}"
    return text


def _present(run_dir):
    """Build every output for one run from its saved metadata."""
    meta = _read_meta(run_dir)
    files = {name: Path(path) for name, path in meta["files"].items()}
    gallery = [(str(files[key]), label) for key, label in INTERMEDIATES.items()]
    downloads = [str(files["glb"])] if "glb" in files else []
    if "obj" in files:
        downloads += [str(path) for path in sorted(files["obj"].parent.iterdir())]
    preview = str(files["glb"] if "glb" in files else files["obj"])
    details = {key: meta[key] for key in DETAIL_KEYS}
    return preview, downloads, gallery, details, _badge(meta), str(files["07_sketch_match"])


def _guarded(call):
    try:
        return call()
    except Sketch2ShapeError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        logging.exception("unexpected failure")
        raise gr.Error(f"Unexpected error: {type(exc).__name__}: {exc}") from exc


def _generate(cfg, mode, drawn, uploaded, prompt, seed, steps, scale, progress):
    sketch = _pick_sketch(mode, drawn, uploaded)

    def on_stage(name):
        progress((pipeline.STAGES.index(name) + 1) / (len(pipeline.STAGES) + 1), desc=name.replace("_", " "))

    result = _guarded(lambda: pipeline.run(sketch, prompt, _overrides(seed, steps, scale), cfg, on_stage))
    for warning in result.warnings:
        gr.Warning(warning)
    return _present(result.run_dir)


def _generate_variants(cfg, mode, drawn, uploaded, prompt, seed, steps, scale, progress):
    sketch = _pick_sketch(mode, drawn, uploaded)
    count = cfg["variants"]["count"]
    stages = len(pipeline.STAGES)

    def on_stage(number, name):
        done = (number - 1 + (pipeline.STAGES.index(name) + 1) / (stages + 1)) / count
        progress(done, desc=f"variant {number} of {count}: {name.replace('_', ' ')}")

    results, errors = _guarded(
        lambda: pipeline.run_variants(sketch, prompt, _overrides(seed, steps, scale), cfg, on_stage)
    )
    for error in errors:
        gr.Warning(f"One variant failed: {error}")
    captions = [
        (str(result.files["07_sketch_match"]),
         f"#{rank} · {result.metrics['mesh_ink_coverage']:.0%} match · seed {result.metrics['seed']}")
        for rank, result in enumerate(results, start=1)
    ]
    runs = [str(result.run_dir) for result in results]
    return (captions, runs, *_present(runs[0]), results[0].metrics["seed"])


def _select_variant(runs, event: gr.SelectData):
    """Show the chosen variant and lock its seed, so Generate can start from it."""
    run_dir = runs[event.index]
    return (*_present(run_dir), _read_meta(run_dir)["metrics"]["seed"])


def build_app(cfg):
    settings = cfg["app"]
    generate_cfg = cfg["sketch_to_image"]
    steps_low, steps_high = settings["steps_range"]
    scale_low, scale_high = settings["controlnet_scale_range"]

    with gr.Blocks(title="Sketch2Shape", analytics_enabled=settings["analytics"]) as app:
        gr.Markdown("# Sketch2Shape\nDraw or upload a sketch, then press Generate. The prompt is optional.")
        mode = gr.State(DRAW)
        runs = gr.State([])
        with gr.Row():
            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("Draw") as draw_tab:
                        drawn = gr.Sketchpad(
                            type="numpy", image_mode="RGBA", canvas_size=tuple(settings["canvas_size"]),
                            brush=gr.Brush(colors=["#000000"], default_size=settings["brush_size"]),
                            eraser=gr.Eraser(), label="Sketch",
                        )
                    with gr.Tab("Upload") as upload_tab:
                        uploaded = gr.Image(type="pil", image_mode="RGBA", sources=["upload"], label="Sketch image")
                prompt = gr.Textbox(label="Prompt (optional)", placeholder="object type, material, color, style")
                with gr.Accordion("Advanced settings", open=False):
                    seed = gr.Number(label="Seed", value=cfg["runtime"]["seed"], precision=0)
                    steps = gr.Slider(steps_low, steps_high, value=generate_cfg["steps"], step=1, label="Steps")
                    scale = gr.Slider(
                        scale_low, scale_high, value=generate_cfg["controlnet_scale"],
                        step=settings["controlnet_scale_step"], label="Sketch faithfulness",
                    )
                with gr.Row():
                    button = gr.Button("Generate", variant="primary")
                    variants_button = gr.Button(f"Generate {cfg['variants']['count']} variants (slower)")
            with gr.Column():
                badge = gr.Markdown()
                model = gr.Model3D(label="3D preview")
                match_view = gr.Image(label="Sketch match: green strokes are on the model, red missed", interactive=False)
                variants = gr.Gallery(label="Variants, best first. Click one to view it and lock its seed", columns=3)
                files = gr.File(label="Download", file_count="multiple")
                with gr.Accordion("Details", open=False):
                    gallery = gr.Gallery(label="Intermediate images", columns=3)
                    details = gr.JSON(label="Metrics, timings and validation")

        draw_tab.select(lambda: DRAW, outputs=mode)
        upload_tab.select(lambda: UPLOAD, outputs=mode)
        inputs = [mode, drawn, uploaded, prompt, seed, steps, scale]
        shown = [model, files, gallery, details, badge, match_view]

        def generate(*args, progress=gr.Progress()):
            return _generate(cfg, *args, progress)

        def generate_variants(*args, progress=gr.Progress()):
            return _generate_variants(cfg, *args, progress)

        button.click(generate, inputs=inputs, outputs=shown)
        variants_button.click(generate_variants, inputs=inputs, outputs=[variants, runs, *shown, seed])
        variants.select(_select_variant, inputs=runs, outputs=[*shown, seed])
    return app


def main():
    parser = argparse.ArgumentParser(description="Start the Sketch2Shape app")
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config, args.set)
    settings = cfg["app"]
    outputs = resolve_path(cfg["paths"]["outputs_dir"])
    outputs.mkdir(parents=True, exist_ok=True)
    # One run at a time: long jobs wait in the queue and the GPU is never shared
    build_app(cfg).queue(default_concurrency_limit=1).launch(
        server_name=settings["host"], server_port=settings["port"], share=settings["share"],
        allowed_paths=[str(outputs)],
    )


if __name__ == "__main__":
    main()
