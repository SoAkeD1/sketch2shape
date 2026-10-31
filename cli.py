"""Run the pipeline from the terminal. Same outputs as the app.

python cli.py --input sketch.png [--prompt "a red chair"] [--variants] [--config config.yaml] [--set section.key=value]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sketch2shape import pipeline  # noqa: E402
from sketch2shape.config import load_config  # noqa: E402
from sketch2shape.errors import Sketch2ShapeError  # noqa: E402


def _summary(result, cfg):
    return {
        "run_dir": str(result.run_dir),
        "files": {fmt: str(result.files[fmt]) for fmt in cfg["export"]["formats"]},
        "subject": result.subject,
        "textured": result.textured,
        "seconds": round(sum(stage["seconds"] for stage in result.stages), 2),
        "metrics": result.metrics,
        "attempts": result.attempts,
        "warnings": result.warnings,
        "notes": result.notes,
    }


def main():
    parser = argparse.ArgumentParser(description="Turn a sketch into a textured 3D model")
    parser.add_argument("--input", required=True, help="path to the sketch image")
    parser.add_argument("--prompt", help="optional: object type, material, color, style")
    parser.add_argument("--variants", action="store_true", help="generate several models, best first")
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        cfg = load_config(args.config, args.set)
        if args.variants:
            results, errors = pipeline.run_variants(args.input, args.prompt, cfg=cfg)
            output = {"variants": [_summary(result, cfg) for result in results], "failed": errors}
        else:
            output = _summary(pipeline.run(args.input, args.prompt, cfg=cfg), cfg)
    except Sketch2ShapeError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
