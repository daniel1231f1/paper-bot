"""Atomic JSON persistence. A crashed or cancelled run can never leave a half-written file:
data goes to a temp file in the same directory, is fsynced, then renamed over the target
(os.replace is atomic on POSIX and Windows)."""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("bot.state")

SCHEMA = 2


def sanitize(obj: Any) -> Any:
    """Recursively convert to strict JSON types: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "item"):  # numpy scalar
        return sanitize(obj.item())
    return str(obj)


class StateError(RuntimeError):
    """State file exists but cannot be trusted. The bot refuses to trade rather than guess."""


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(sanitize(payload), fh, separators=(",", ":"), allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class Store:
    """state.json = resumable engine state; dashboard.json = derived view for the static site."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.state_path = directory / "state.json"
        self.dashboard_path = directory / "dashboard.json"

    def load(self) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        try:
            data = json.loads(self.state_path.read_text())
        except (OSError, ValueError) as exc:
            raise StateError(f"state file is unreadable: {exc}") from exc
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            return None  # state from an older bot version: start a fresh experiment
        for key in ("config", "config_hash", "engine"):
            if key not in data:
                raise StateError(f"state file is missing '{key}'")
        return data

    def save(self, state: dict[str, Any]) -> None:
        state["schema"] = SCHEMA
        atomic_write_json(self.state_path, state)

    def save_dashboard(self, view: dict[str, Any]) -> None:
        atomic_write_json(self.dashboard_path, view)
