"""Safe JSON and artifact helpers shared by experiments and Flask."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Recursively convert NumPy/Path values and non-finite numbers for JSON."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic, human-readable JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON, returning ``default`` only when the artifact is absent."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
