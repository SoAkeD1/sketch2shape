"""Summarize saved runs as a Markdown table of mean ± std, from each run_meta.json.

Usage: python scripts/summarize_metrics.py [--since RUN_ID] [--out docs/METRICS.md]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sketch2shape.config import load_config, resolve_path  # noqa: E402

METRIC_KEYS = [
    ("ink_coverage", "Sketch fidelity: ink coverage"),
    ("mesh_ink_coverage", "Sketch fidelity: ink coverage on the mesh"),
    ("silhouette_iou", "Sketch fidelity: silhouette IoU"),
    ("image_iou", "Sketch fidelity: generated image IoU"),
    ("clip_sketch_to_views", "Semantic: CLIP sketch to views"),
    ("clip_text_to_views", "Semantic: CLIP text to views"),
    ("clip_view_consistency", "Multi-view consistency: CLIP"),
    ("degenerate_face_ratio", "Geometry: degenerate face ratio"),
    ("bbox_aspect_ratio", "Geometry: bounding box aspect ratio"),
    ("faces", "Geometry: faces"),
]


def _row(label, values):
    values = [v for v in values if v is not None]
    if not values:
        return f"| {label} | n/a | 0 |"
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"| {label} | {statistics.mean(values):.3f} ± {spread:.3f} | {len(values)} |"


def summarize(runs):
    good = [run for run in runs if run.get("status") == "ok"]
    warm = [run for run in good if not run.get("cold_start")]
    cold = [run for run in good if run.get("cold_start")]
    lines = ["| Metric | Mean ± std | Runs |", "|---|---|---|"]
    for key, label in METRIC_KEYS:
        lines.append(_row(label, [run["metrics"].get(key) for run in good]))
    lines.append(_row("Time: total, warm (s)", [run["total_seconds"] for run in warm]))
    lines.append(_row("Time: total, cold (s)", [run["total_seconds"] for run in cold]))
    stages = sorted({stage["stage"] for run in warm for stage in run["stages"]})
    for name in stages:
        per_run = [sum(s["seconds"] for s in run["stages"] if s["stage"] == name) for run in warm]
        lines.append(_row(f"Time: {name}, warm (s)", per_run))
    lines.append(_row("Peak VRAM (GB)", [run["peak_vram_gb"] for run in good]))
    lines.append(_row("Peak RAM (GB)", [run["peak_ram_gb"] for run in good]))
    lines.append(_row("Attempts used", [len(run["attempts"]) for run in good]))
    watertight = sum(1 for run in good if run["metrics"].get("watertight"))
    low = sum(1 for run in good if run.get("warnings"))
    lines += [
        "",
        f"- **Reliability:** {len(good)} of {len(runs)} runs produced a validated model.",
        f"- **Watertight:** {watertight} of {len(good)}.",
        f"- **Flagged with a quality warning:** {low} of {len(good)}.",
    ]
    for run in runs:
        if run.get("status") != "ok":
            lines.append(f"- **Failed run** {run.get('run_id')}: {run.get('error')}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="", help="only runs with an id after this one")
    parser.add_argument("--config")
    parser.add_argument("--out")
    args = parser.parse_args()
    outputs = resolve_path(load_config(args.config)["paths"]["outputs_dir"])
    runs = []
    for path in sorted(outputs.glob("*/run_meta.json")):
        if path.parent.name > args.since:
            with open(path, encoding="utf-8") as handle:
                runs.append(json.load(handle))
    table = summarize(runs)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
