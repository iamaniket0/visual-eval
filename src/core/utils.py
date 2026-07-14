"""Shared utilities: config loading, cost tracking, logging, JSONL I/O."""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent

load_dotenv(ROOT / ".env")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def get_api_key(env_var: str) -> str | None:
    """Fetch API key from env. Returns None if not set (scaffold-friendly)."""
    val = os.getenv(env_var)
    return val if val else None


@dataclass
class CostTracker:
    """Thread-safe cost tracker with hard cap enforcement."""
    hard_cap_usd: float
    alert_at_fraction: float = 0.8
    total: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)
    by_stage: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _alerted: bool = False

    def add(self, amount: float, model: str = "unknown", stage: str = "unknown") -> None:
        with self._lock:
            self.total += amount
            self.by_model[model] = self.by_model.get(model, 0.0) + amount
            self.by_stage[stage] = self.by_stage.get(stage, 0.0) + amount
            if (not self._alerted
                    and self.total >= self.hard_cap_usd * self.alert_at_fraction):
                self._alerted = True
                logging.getLogger("cost").warning(
                    "Cost at %.0f%% of cap: $%.2f / $%.2f",
                    100 * self.total / self.hard_cap_usd,
                    self.total, self.hard_cap_usd,
                )

    def check_cap(self) -> bool:
        """Return True if we're still under cap."""
        return self.total < self.hard_cap_usd

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_usd": round(self.total, 4),
                "cap_usd": self.hard_cap_usd,
                "by_model": {k: round(v, 4) for k, v in self.by_model.items()},
                "by_stage": {k: round(v, 4) for k, v in self.by_stage.items()},
            }


_JSONL_LOCKS: dict[str, threading.Lock] = {}
_JSONL_LOCKS_LOCK = threading.Lock()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path)
    with _JSONL_LOCKS_LOCK:
        if key not in _JSONL_LOCKS:
            _JSONL_LOCKS[key] = threading.Lock()
        lock = _JSONL_LOCKS[key]
    with lock, open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
