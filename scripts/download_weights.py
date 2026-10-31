"""Download model weights listed in config.yaml. Re-running resumes partial files.

Usage: python scripts/download_weights.py [--only name ...] [--dry-run] [--set key=value]
"""
import argparse
import fnmatch
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import HfApi, snapshot_download  # noqa: E402
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError  # noqa: E402

from sketch2shape.config import load_config, resolve_path  # noqa: E402
from sketch2shape.errors import ConfigError, WeightsError  # noqa: E402


COMMIT_STAMP = ".commit"


def _matches(filename, patterns):
    return not patterns or any(fnmatch.fnmatch(filename, p) for p in patterns)


def repo_size(repo_id, patterns):
    """Return (bytes, file count) for the files the patterns select."""
    info = HfApi().model_info(repo_id, files_metadata=True)
    selected = [s for s in info.siblings if _matches(s.rfilename, patterns)]
    return sum(s.size or 0 for s in selected), len(selected)


def fetch_third_party(cfg):
    """Fetch source-only dependencies as a zip of the commit pinned in config."""
    root = resolve_path(cfg["paths"]["third_party_dir"])
    for name, entry in cfg["third_party"].items():
        target = root / name
        stamp = target / COMMIT_STAMP
        if stamp.is_file() and stamp.read_text().strip() == entry["commit"]:
            print(f"{name}: already at {entry['commit']}")
            continue
        url = f"{entry['repo_url']}/archive/{entry['commit']}.zip"
        try:
            with urllib.request.urlopen(url) as response:
                archive = zipfile.ZipFile(io.BytesIO(response.read()))
        except OSError as exc:
            raise WeightsError(f"{name}: download of {url} failed: {exc}") from exc
        if target.exists():
            shutil.rmtree(target)
        for info in archive.infolist():
            # Every entry sits under one top folder named after the commit
            relative = info.filename.split("/", 1)[1]
            if not relative or info.is_dir():
                continue
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(info))
        stamp.write_text(entry["commit"])
        print(f"{name}: {entry['repo_url']} @ {entry['commit']}")


def fetch_rembg_model(cfg):
    """rembg keeps its own model cache and downloads on first use, so trigger that now."""
    from rembg import new_session

    model = cfg["background"]["rembg_model"]
    new_session(model)
    print(f"rembg: {model} ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--only", nargs="*", help="weight names from config.weights")
    parser.add_argument("--dry-run", action="store_true", help="report sizes without downloading")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    entries = cfg["weights"]
    unknown = set(args.only or []) - set(entries)
    if unknown:
        raise ConfigError(f"unknown weight names: {sorted(unknown)}")
    names = args.only or list(entries)
    weights_dir = resolve_path(cfg["paths"]["weights_dir"])
    bytes_per_gb = cfg["runtime"]["bytes_per_gb"]

    total = 0
    for name in names:
        repo_id = entries[name]["repo_id"]
        patterns = entries[name].get("allow_patterns")
        try:
            size, count = repo_size(repo_id, patterns)
            if count == 0:
                raise WeightsError(f"{name}: no files in {repo_id} match {patterns}")
            total += size
            print(f"{name}: {repo_id} | {count} files | {size / bytes_per_gb:.2f} GB")
            if not args.dry_run:
                local_dir = None if entries[name].get("use_hf_cache") else weights_dir / name
                snapshot_download(repo_id, allow_patterns=patterns, local_dir=local_dir)
        except GatedRepoError as exc:
            raise WeightsError(f"{name}: {repo_id} is gated", hint="accept its license on Hugging Face and log in") from exc
        except HfHubHTTPError as exc:
            raise WeightsError(f"{name}: download of {repo_id} failed: {exc}") from exc
    print(f"TOTAL: {total / bytes_per_gb:.2f} GB")
    if not args.dry_run and not args.only:
        fetch_third_party(cfg)
        fetch_rembg_model(cfg)


if __name__ == "__main__":
    main()
