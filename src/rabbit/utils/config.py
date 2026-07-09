"""YAML config loader with dotted-key overrides and ${ENV:VAR} expansion.

RABBiT configs are plain YAML. To keep them portable we resolve two things at
load time:

  * ``${VAR}`` and ``${VAR:fallback}`` references → environment variables.
  * ``"data.subjects=[1,2,3]"`` style CLI overrides → dotted-path assignments,
    coerced to int/float/bool when the literal allows.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _expand_env_strings(obj: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:fallback}`` in string leaves."""
    if isinstance(obj, str):
        def _sub(match: re.Match) -> str:
            name = match.group(1)
            fallback = match.group(2)
            value = os.environ.get(name)
            if value is not None:
                return value
            if fallback is not None:
                return fallback
            raise KeyError(
                f"Environment variable {name!r} not set and no fallback given in '{obj}'."
            )
        return _ENV_PATTERN.sub(_sub, obj)
    if isinstance(obj, dict):
        return {k: _expand_env_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_strings(v) for v in obj]
    return obj


def _coerce_scalar(value: str) -> Any:
    """Best-effort scalar coercion for CLI override values."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _apply_override(config: dict, dotted_key: str, value: str) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = _coerce_scalar(value)


def load_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
    expand_env: bool = True,
) -> dict:
    """Load a RABBiT YAML config.

    Args:
        path: filesystem path to the YAML config.
        overrides: iterable of ``"dotted.key=value"`` strings.
        expand_env: when True, expand ``${VAR}`` / ``${VAR:default}`` references
            in string leaves.
    """
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    if expand_env:
        config = _expand_env_strings(config)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be 'key=value', got: {ov!r}")
        key, _, value = ov.partition("=")
        _apply_override(config, key.strip(), value.strip())
    return config
