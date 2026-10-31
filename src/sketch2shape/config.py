"""Config loader. Precedence: config.yaml < S2S_* env vars < CLI overrides."""
import copy
import os
from pathlib import Path

import yaml

from .errors import ConfigError

ENV_PREFIX = "S2S_"
ENV_SEPARATOR = "__"
CONFIG_PATH_ENV = "S2S_CONFIG"
DEFAULT_CONFIG_NAME = "config.yaml"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_value(raw):
    """Parse an override string with YAML rules, so 30 -> int and true -> bool."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse override value {raw!r}: {exc}") from exc


def _set_path(cfg, keys, value, source):
    node = cfg
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            raise ConfigError(f"unknown config section {'.'.join(keys)} from {source}")
        node = node[key]
    if keys[-1] not in node:
        raise ConfigError(
            f"unknown config key {'.'.join(keys)} from {source}",
            hint="check the key names in config.yaml",
        )
    node[keys[-1]] = value


def _apply_env(cfg, environ):
    for name, raw in environ.items():
        if name == CONFIG_PATH_ENV or not name.startswith(ENV_PREFIX):
            continue
        keys = name[len(ENV_PREFIX):].lower().split(ENV_SEPARATOR)
        _set_path(cfg, keys, _parse_value(raw), f"env var {name}")


def apply_overrides(cfg, overrides):
    """Apply 'section.key=value' strings and return a new config."""
    result = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like section.key=value")
        path, raw = item.split("=", 1)
        _set_path(result, path.strip().split("."), _parse_value(raw), f"override {item!r}")
    return result


def load_config(path=None, overrides=None, environ=None):
    path = Path(path or os.environ.get(CONFIG_PATH_ENV) or REPO_ROOT / DEFAULT_CONFIG_NAME)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path} must contain a mapping")
    _apply_env(cfg, os.environ if environ is None else environ)
    return apply_overrides(cfg, overrides)


def resolve_path(relative):
    """Resolve a configured path against the repo root."""
    path = Path(relative)
    return path if path.is_absolute() else REPO_ROOT / path
