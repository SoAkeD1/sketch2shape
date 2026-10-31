"""Gradio front end in the Living Notebook design. All work happens in pipeline.py."""
import argparse
import json
import logging
import queue
import random
import sys
import threading
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr  # noqa: E402

from sketch2shape import pipeline, ui_html  # noqa: E402
from sketch2shape.config import load_config, resolve_path  # noqa: E402
from sketch2shape.errors import Sketch2ShapeError  # noqa: E402

DRAW, UPLOAD = "draw", "upload"
OBJ_ZIP_NAME = "model_obj.zip"
INTERMEDIATES = {
    "01_clean_sketch": "1. Clean sketch",
    "02_control_map": "2. Control map",
    "03_sketch_mask": "3. Sketch mask",
    "04_generated": "4. Generated image",
    "05_foreground": "5. Foreground",
    "06_views": "6. Mesh from all sides",
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


def _obj_zip(run_dir, files):
    """Keep the .obj, .mtl and texture together in one download."""
    if "obj" not in files:
        return None
    target = Path(run_dir) / OBJ_ZIP_NAME
    if not target.exists():
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(files["obj"].parent.iterdir()):
                archive.write(path, path.name)
    return target


def _present(run_dir, cfg):
    """Build every result output for one run from its saved metadata."""
    meta = _read_meta(run_dir)
    files = {name: Path(path) for name, path in meta["files"].items()}
    gallery = [(str(files[key]), label) for key, label in INTERMEDIATES.items()]
    zip_path = _obj_zip(run_dir, files)
    downloads = [str(files["glb"])] if "glb" in files else []
    if "obj" in files:
        downloads += [str(path) for path in sorted(files["obj"].parent.iterdir())]
    if zip_path:
        downloads.append(str(zip_path))
    preview = str(files["glb"] if "glb" in files else files["obj"])
    return {
        "model": preview,
        "files": downloads,
        "gallery": gallery,
        "details": {key: meta[key] for key in DETAIL_KEYS},
        "badge": ui_html.score_badge(meta),
        "angles": ui_html.angle_strip(files["06_views"], cfg["metrics"]["n_views"]),
        "match": ui_html.match_panel(meta["run_id"], files),
        "dl_panel": ui_html.downloads_panel(files, zip_path),
        "quality": ui_html.quality_panel(meta),
        "performance": ui_html.performance_panel(meta),
        "checks": ui_html.checks_panel(meta),
        "geometry": ui_html.geometry_panel(meta),
    }


def _stream(work):
    """Run the pipeline on a worker thread and hand its events to the generator as they happen."""
    events = queue.Queue()

    def target():
        try:
            events.put(("done", work(events.put)))
        except Exception as exc:  # Passed to the UI thread, which turns it into a visible error
            events.put(("failed", exc))

    threading.Thread(target=target, daemon=True).start()
    while True:
        kind, payload = events.get()
        yield kind, payload
        if kind in ("done", "failed"):
            return


def _raise_for(exc):
    if isinstance(exc, Sketch2ShapeError):
        raise gr.Error(str(exc)) from exc
    logging.error("unexpected failure", exc_info=exc)
    raise gr.Error(f"Unexpected error: {type(exc).__name__}: {exc}") from exc


def build_app(cfg):
    settings = cfg["app"]
    generate_cfg = cfg["sketch_to_image"]
    steps_low, steps_high = settings["steps_range"]
    scale_low, scale_high = settings["controlnet_scale_range"]
    variant_count = cfg["variants"]["count"]
    variant_minutes = variant_count * settings["variant_estimate_seconds"] / 60

    with gr.Blocks(title="Sketch2Shape", analytics_enabled=settings["analytics"]) as app:
        gr.HTML(ui_html.header())
        mode = gr.State(DRAW)
        runs = gr.State([])
        latest_model = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=settings["column_min_width"]):
                with gr.Tabs():
                    with gr.Tab("Draw") as draw_tab:
                        drawn = gr.Sketchpad(
                            type="numpy", image_mode="RGBA", canvas_size=tuple(settings["canvas_size"]),
                            height=settings["canvas_height"],
                            brush=gr.Brush(colors=["#000000"], default_size=settings["brush_size"]),
                            eraser=gr.Eraser(), label="Your sketch", elem_id="nb-canvas",
                        )
                    with gr.Tab("Upload") as upload_tab:
                        uploaded = gr.Image(
                            type="pil", image_mode="RGBA", sources=["upload"],
                            label="Sketch image: dark lines on light paper work best",
                        )
                prompt = gr.Textbox(
                    label="Describe it (optional)", placeholder="e.g. a red ceramic mug",
                    info="Leave it empty and it will guess from your drawing. Words change color and material, and can change the shape too.",
                )
                with gr.Row(elem_classes="nb-chip-row"):
                    chips = [gr.Button(f"+ {chip}", size="sm") for chip in settings["prompt_chips"]]
                with gr.Accordion("Advanced settings", open=False):
                    seed = gr.Number(
                        label="Seed", value=cfg["runtime"]["seed"], precision=0,
                        info="Same number, same starting point.",
                    )
                    steps = gr.Slider(
                        steps_low, steps_high, value=generate_cfg["steps"], step=1, label="Detail (steps)",
                        info="Faster ↔ More detail",
                    )
                    scale = gr.Slider(
                        scale_low, scale_high, value=generate_cfg["controlnet_scale"],
                        step=settings["controlnet_scale_step"], label="Sketch faithfulness",
                        info="More creative ↔ Follow my lines",
                    )
                    with gr.Row():
                        shuffle = gr.Button("Shuffle seed", size="sm", elem_classes="nb-small-btn")
                        reset_settings = gr.Button("Reset to defaults", size="sm", elem_classes="nb-small-btn")
                with gr.Row():
                    button = gr.Button("Generate 3D model", variant="primary", elem_id="nb-generate", scale=3)
                    variants_button = gr.Button(
                        f"Generate {variant_count} variants · about {variant_minutes:.1f} min",
                        elem_id="nb-variants", scale=2,
                    )
                new_sketch = gr.Button("Start a new sketch", size="sm", elem_classes="nb-small-btn")
            with gr.Column(scale=4, min_width=settings["column_min_width"]):
                empty = gr.HTML(ui_html.empty_panel())
                run_panel = gr.HTML(visible=False)
                badge = gr.HTML()
                # Stays on the page and is hidden by CSS while empty: the viewer stays blank if it
                # first appears inside a container that was hidden
                model = gr.Model3D(
                    label="3D preview: drag to turn it, scroll to zoom", height=settings["model_height"],
                    elem_id="nb-model",
                )
                with gr.Column(visible=False) as result_box:
                    angles = gr.HTML()
                    match_view = gr.HTML()
                    with gr.Column(visible=False) as variants_box:
                        variants = gr.Gallery(
                            label="Variants, best match first. Ranked by how well they follow your lines, not by looks, so pick the one you like. Clicking one locks its seed.",
                            columns=variant_count, elem_id="nb-variant-gallery", allow_preview=False,
                        )
                    dl_panel = gr.HTML()
                    with gr.Accordion("All files", open=False):
                        files = gr.File(label="Download", file_count="multiple")
                    with gr.Accordion("Details: how this model was made", open=False):
                        with gr.Tabs():
                            with gr.Tab("Journey"):
                                gallery = gr.Gallery(label="From sketch to model", columns=3)
                            with gr.Tab("Quality"):
                                quality = gr.HTML()
                            with gr.Tab("Performance"):
                                performance = gr.HTML()
                            with gr.Tab("Checks"):
                                checks = gr.HTML()
                            with gr.Tab("Geometry"):
                                geometry = gr.HTML()
                            with gr.Tab("Raw data"):
                                details = gr.JSON(label="Metrics, timings and validation")

        result_parts = {
            "files": files, "gallery": gallery, "details": details, "badge": badge, "angles": angles,
            "match": match_view, "dl_panel": dl_panel, "quality": quality, "performance": performance,
            "checks": checks, "geometry": geometry,
        }
        panels = [*result_parts.values(), run_panel, empty, result_box, variants_box, variants, runs, seed]
        stream_outputs = [*panels, latest_model]

        def show_result(run_dir, extra=None, streaming=True):
            built = _present(run_dir, cfg)
            preview = built.pop("model")
            update = {result_parts[key]: value for key, value in built.items()}
            # The 3D viewer stays blank if a generator sends it the model, so a plain step does that afterwards
            update[latest_model if streaming else model] = preview
            update.update({run_panel: gr.update(visible=False), empty: gr.update(visible=False),
                           result_box: gr.update(visible=True)})
            update.update(extra or {})
            return update

        def show_running(html):
            return {run_panel: gr.update(value=html, visible=True), empty: gr.update(visible=False),
                    result_box: gr.update(visible=False), badge: ""}

        def show_failed():
            return {run_panel: gr.update(visible=False), empty: gr.update(visible=True),
                    result_box: gr.update(visible=False), badge: ""}

        def generate(mode_value, drawn_value, uploaded_value, prompt_value, seed_value, steps_value, scale_value):
            sketch = _pick_sketch(mode_value, drawn_value, uploaded_value)
            has_prompt = bool((prompt_value or "").strip())
            tracker = ui_html.RunTracker(pipeline.STAGES, has_prompt, cfg["retry"]["max_attempts"])
            yield show_running(tracker.render())

            def work(emit):
                return pipeline.run(
                    sketch, prompt_value, _overrides(seed_value, steps_value, scale_value), cfg,
                    on_stage=lambda name: emit(("stage", name)),
                    on_artifact=lambda stem, path: emit(("image", (stem, path))),
                )

            for kind, payload in _stream(work):
                if kind == "stage":
                    tracker.stage(payload)
                    yield show_running(tracker.render())
                elif kind == "image":
                    tracker.image(*payload)
                    yield show_running(tracker.render())
                elif kind == "failed":
                    yield show_failed()
                    _raise_for(payload)
                else:
                    for warning in payload.warnings:
                        gr.Warning(warning)
                    yield show_result(payload.run_dir, {variants_box: gr.update(visible=False), runs: []})

        def generate_variants(mode_value, drawn_value, uploaded_value, prompt_value, seed_value, steps_value, scale_value):
            sketch = _pick_sketch(mode_value, drawn_value, uploaded_value)
            has_prompt = bool((prompt_value or "").strip())
            tracker = ui_html.RunTracker(
                pipeline.STAGES, has_prompt, cfg["variants"]["max_attempts"], variant_total=variant_count
            )
            yield show_running(tracker.render())

            def work(emit):
                return pipeline.run_variants(
                    sketch, prompt_value, _overrides(seed_value, steps_value, scale_value), cfg,
                    on_stage=lambda number, name: emit(("stage", (number, name))),
                    on_artifact=lambda number, stem, path: emit(("image", (number, stem, path))),
                )

            for kind, payload in _stream(work):
                if kind == "stage":
                    tracker.begin_variant(payload[0])
                    tracker.stage(payload[1])
                    yield show_running(tracker.render())
                elif kind == "image":
                    tracker.begin_variant(payload[0])
                    tracker.image(payload[1], payload[2])
                    yield show_running(tracker.render())
                elif kind == "failed":
                    yield show_failed()
                    _raise_for(payload)
                else:
                    results, errors = payload
                    for error in errors:
                        gr.Warning(f"One variant failed: {error}")
                    captions = [
                        (str(result.files["07_sketch_match"]),
                         f"#{rank} · {result.metrics['mesh_ink_coverage']:.0%} match · seed {result.metrics['seed']}")
                        for rank, result in enumerate(results, start=1)
                    ]
                    yield show_result(results[0].run_dir, {
                        variants_box: gr.update(visible=True), variants: captions,
                        runs: [str(result.run_dir) for result in results], seed: results[0].metrics["seed"],
                    })

        def select_variant(run_dirs, event: gr.SelectData):
            """Show the chosen variant and lock its seed, so Generate starts from it."""
            run_dir = run_dirs[event.index]
            return show_result(run_dir, {seed: _read_meta(run_dir)["metrics"]["seed"]}, streaming=False)

        def start_new():
            cleared = {part: None for part in (model, latest_model, files, gallery, details)}
            cleared.update({part: "" for part in (angles, match_view, dl_panel, quality, performance, checks, geometry)})
            cleared.update(show_failed())
            cleared.update({variants_box: gr.update(visible=False), variants: None, runs: [],
                            uploaded: None, prompt: ""})
            return cleared

        draw_tab.select(lambda: DRAW, outputs=mode)
        upload_tab.select(lambda: UPLOAD, outputs=mode)
        inputs = [mode, drawn, uploaded, prompt, seed, steps, scale]
        for trigger, work in ((button, generate), (variants_button, generate_variants)):
            trigger.click(
                lambda: (None, None), outputs=[model, latest_model], show_progress="hidden",
            ).then(
                work, inputs=inputs, outputs=stream_outputs, show_progress="hidden",
            ).then(
                lambda preview: preview, inputs=latest_model, outputs=model, show_progress="hidden",
            )
        variants.select(select_variant, inputs=runs, outputs=[*panels, model], show_progress="hidden")
        for chip_button, chip in zip(chips, settings["prompt_chips"]):
            chip_button.click(
                lambda text, word=chip: f"{text}, {word}" if (text or "").strip() else word,
                inputs=prompt, outputs=prompt, show_progress="hidden",
            )
        shuffle.click(lambda: random.randint(0, settings["seed_max"]), outputs=seed, show_progress="hidden")
        reset_settings.click(
            lambda: (cfg["runtime"]["seed"], generate_cfg["steps"], generate_cfg["controlnet_scale"]),
            outputs=[seed, steps, scale], show_progress="hidden",
        )
        new_sketch.click(
            start_new, outputs=[*stream_outputs, model, uploaded, prompt], show_progress="hidden",
            js="() => { window.nbResetCanvas && window.nbResetCanvas(); return []; }",
        )
    return app


def launch_options(cfg):
    """Theme files and fonts. If a font can't load, the system font takes over."""
    settings = cfg["app"]
    ui_dir = resolve_path(cfg["paths"]["ui_dir"])
    script = (ui_dir / "notebook.js").read_text(encoding="utf-8")
    return {
        "theme": gr.themes.Base(font=["Inter", "system-ui", "sans-serif"], font_mono=["JetBrains Mono", "monospace"]),
        "css_paths": [str(ui_dir / "notebook.css")],
        "head": f'<link rel="stylesheet" href="{settings["fonts_url"]}"><script>{script}</script>',
    }


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
        allowed_paths=[str(outputs)], **launch_options(cfg),
    )


if __name__ == "__main__":
    main()
