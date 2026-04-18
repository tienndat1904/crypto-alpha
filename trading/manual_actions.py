"""
Manual Actions Queue
====================
Cross-process queue for user-initiated actions from the dashboard.

Producer: dashboard.py (appends actions)
Consumer: paper_trader.py / futures_trader.py (poll + execute + remove)

Single state.json writer remains the bot — dashboard never mutates state directly.
Atomic file writes via temp file + rename.
"""

import json
import os
import time
import uuid
from pathlib import Path

QUEUE_FILE = Path("trading/manual_actions.json")


def _read_all() -> list:
    if not QUEUE_FILE.exists():
        return []
    try:
        with open(QUEUE_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _atomic_write(actions: list) -> None:
    QUEUE_FILE.parent.mkdir(exist_ok=True)
    tmp = QUEUE_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(actions, f, indent=2)
    os.replace(tmp, QUEUE_FILE)


def append_close(mode: str, symbol: str, pct: float) -> str:
    """Queue a close action. mode='spot' or 'futures'. pct in (0, 1]."""
    pct = max(0.01, min(1.0, float(pct)))
    action = {
        "id": uuid.uuid4().hex[:12],
        "type": "close",
        "mode": mode,
        "symbol": symbol,
        "pct": pct,
        "ts": time.time(),
        "status": "pending",
    }
    actions = _read_all()
    actions.append(action)
    _atomic_write(actions)
    return action["id"]


def pending_for(mode: str, symbol: str = None) -> list:
    """Return pending actions matching mode (and optionally symbol)."""
    out = []
    for a in _read_all():
        if a.get("status") != "pending":
            continue
        if a.get("mode") != mode:
            continue
        if symbol and a.get("symbol") != symbol:
            continue
        out.append(a)
    return out


def consume(mode: str) -> list:
    """Atomically remove and return all pending actions for the given mode."""
    actions = _read_all()
    mine, rest = [], []
    for a in actions:
        if a.get("status") == "pending" and a.get("mode") == mode:
            mine.append(a)
        else:
            rest.append(a)
    if mine:
        _atomic_write(rest)
    return mine
