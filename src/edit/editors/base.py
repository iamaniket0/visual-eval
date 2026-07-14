"""Base class for all image editors.

Key difference from T2I eval's BaseGenerator: editors take a SOURCE IMAGE
plus an edit instruction and produce an EDITED image. For multi-turn prompts,
the output of turn N feeds as source to turn N+1.

Supports two API patterns:
    1. sync        - POST returns the edited image directly
    2. async_poll  - POST returns a job id; GET polls until ready

All editors share:
    - Exponential backoff retry on 429/5xx
    - Content filter detection (logged as FILTERED, not retried)
    - Cost accounting via CostTracker
    - PNG normalization on save
    - Source image validation before API call
"""

from __future__ import annotations

import asyncio
import base64
import io
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from src.core.utils import CostTracker, append_jsonl, get_api_key, get_logger
from src.edit import OUTPUTS_DIR


class EditStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FILTERED = "FILTERED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass
class EditResult:
    prompt_id: str
    model: str
    status: EditStatus
    image_path: str | None = None
    source_image_path: str | None = None
    cost_usd: float = 0.0
    duration_sec: float = 0.0
    error: str | None = None
    turn: int = 1
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


_FILTER_MARKERS = (
    "content policy",
    "content_policy",
    "safety",
    "not allowed",
    "nsfw",
    "moderation",
    "violates",
    "blocked",
    "inappropriate",
    "unsafe",
    "prohibited",
    "sensitive_content",
    "safety_system",
)


def looks_like_filter(text: str) -> bool:
    t = text.lower()
    return any(marker in t for marker in _FILTER_MARKERS)


class BaseEditor(ABC):
    """Abstract base for image editing providers."""

    model_id: str = ""
    provider: str = ""

    def __init__(self, config: dict[str, Any], cost_tracker: CostTracker, concurrency: int = 4):
        self.config = config
        self.cost_tracker = cost_tracker
        self.log = get_logger(f"edit.{self.model_id}")
        self.api_key = get_api_key(config["api_key_env"])
        self.cost_per_edit = float(config["cost_per_edit"])
        self.pattern = config.get("pattern", "sync")
        self.supports_mask = config.get("supports_mask", False)
        self.supports_multi_turn = config.get("supports_multi_turn", False)
        self.semaphore = asyncio.Semaphore(concurrency)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0))
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Editor must be used inside `async with`.")
        return self._client

    # ------------------------------------------------------------------
    # Public entry point — single-turn edit
    # ------------------------------------------------------------------

    async def edit(
        self,
        prompt_id: str,
        source_image_path: str,
        edit_instruction: str,
        output_dir: Path,
        mask_path: str | None = None,
        turn: int = 1,
    ) -> EditResult:
        started = datetime.now(timezone.utc)

        if not self.api_key:
            return EditResult(
                prompt_id=prompt_id,
                model=self.model_id,
                status=EditStatus.SKIPPED,
                turn=turn,
                source_image_path=source_image_path,
                error=f"{self.config['api_key_env']} not set",
            )

        if not self.cost_tracker.check_cap():
            return EditResult(
                prompt_id=prompt_id,
                model=self.model_id,
                status=EditStatus.SKIPPED,
                turn=turn,
                source_image_path=source_image_path,
                error="Cost cap reached",
            )

        src_path = Path(source_image_path)
        if not src_path.exists():
            return EditResult(
                prompt_id=prompt_id,
                model=self.model_id,
                status=EditStatus.ERROR,
                turn=turn,
                source_image_path=source_image_path,
                error=f"Source image not found: {source_image_path}",
            )

        async with self.semaphore:
            try:
                image_bytes, raw_meta = await self._edit_with_retry(
                    source_image_path, edit_instruction, mask_path
                )
            except _ContentFiltered as e:
                return EditResult(
                    prompt_id=prompt_id,
                    model=self.model_id,
                    status=EditStatus.FILTERED,
                    turn=turn,
                    source_image_path=source_image_path,
                    error=str(e),
                    raw_metadata=e.metadata,
                )
            except Exception as e:
                return EditResult(
                    prompt_id=prompt_id,
                    model=self.model_id,
                    status=EditStatus.ERROR,
                    turn=turn,
                    source_image_path=source_image_path,
                    error=f"{type(e).__name__}: {e}",
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_t{turn}" if turn > 1 else ""
        out_path = output_dir / f"{prompt_id}{suffix}.png"
        self._save_png(image_bytes, out_path)
        self.cost_tracker.add(self.cost_per_edit, model=self.model_id, stage="editing")

        duration = (datetime.now(timezone.utc) - started).total_seconds()
        return EditResult(
            prompt_id=prompt_id,
            model=self.model_id,
            status=EditStatus.SUCCESS,
            image_path=str(out_path),
            source_image_path=source_image_path,
            cost_usd=self.cost_per_edit,
            duration_sec=round(duration, 2),
            raw_metadata=raw_meta,
            turn=turn,
        )

    # ------------------------------------------------------------------
    # Multi-turn edit — chains turns sequentially
    # ------------------------------------------------------------------

    async def edit_multi_turn(
        self,
        prompt_id: str,
        source_image_path: str,
        instructions: list[str],
        output_dir: Path,
    ) -> list[EditResult]:
        results: list[EditResult] = []
        current_source = source_image_path

        for turn_idx, instruction in enumerate(instructions, start=1):
            result = await self.edit(
                prompt_id=prompt_id,
                source_image_path=current_source,
                edit_instruction=instruction,
                output_dir=output_dir,
                turn=turn_idx,
            )
            results.append(result)

            if result.status != EditStatus.SUCCESS:
                self.log.warning(
                    "Multi-turn chain broken at turn %d/%d for %s: %s",
                    turn_idx,
                    len(instructions),
                    prompt_id,
                    result.error,
                )
                break

            current_source = result.image_path

        return results

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------

    async def _edit_with_retry(
        self, source_path: str, instruction: str, mask_path: str | None = None, max_retries: int = 3
    ) -> tuple[bytes, dict]:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await self._do_edit(source_path, instruction, mask_path)
            except _ContentFiltered:
                raise
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                body_snippet = e.response.text[:500] if e.response is not None else ""
                if looks_like_filter(body_snippet):
                    raise _ContentFiltered(
                        f"Content filter: {body_snippet[:200]}", metadata={"status_code": code}
                    ) from e
                if code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    delay = 2**attempt
                    self.log.warning(
                        "HTTP %d, retrying in %ds (%s)", code, delay, body_snippet[:120]
                    )
                    await asyncio.sleep(delay)
                    last_exc = e
                    continue
                raise
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    last_exc = e
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("Retry loop exited without result")

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    async def _do_edit(
        self, source_image_path: str, instruction: str, mask_path: str | None = None
    ) -> tuple[bytes, dict]:
        """Return (edited_image_bytes, raw_metadata). May raise _ContentFiltered."""

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _save_png(image_bytes: bytes, path: Path) -> None:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        img.save(path, format="PNG")

    @staticmethod
    def _b64_to_bytes(data: str) -> bytes:
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        return base64.b64decode(data)

    @staticmethod
    def _image_to_b64(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    async def _download(self, url: str) -> bytes:
        r = await self.client.get(url)
        r.raise_for_status()
        return r.content

    async def _poll_until_ready(
        self,
        url: str,
        headers: dict[str, str],
        poll_interval: float = 2.0,
        max_wait: float = 120.0,
        ready_check=None,
        failed_check=None,
    ) -> dict:
        elapsed = 0.0
        while elapsed < max_wait:
            r = await self.client.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
            if failed_check and (reason := failed_check(data)):
                if looks_like_filter(reason):
                    raise _ContentFiltered(reason, metadata=data)
                raise RuntimeError(f"Job failed: {reason}")
            if ready_check(data):
                return data
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        raise TimeoutError(f"Job did not complete within {max_wait}s")


class _ContentFiltered(Exception):
    def __init__(self, msg: str, metadata: dict | None = None):
        super().__init__(msg)
        self.metadata = metadata or {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseEditor]] = {}


def register(model_id: str):
    def deco(cls):
        cls.model_id = model_id
        _REGISTRY[model_id] = cls
        return cls

    return deco


def get_editor(
    model_id: str, config: dict[str, Any], cost_tracker: CostTracker, concurrency: int = 4
) -> BaseEditor:
    if model_id not in _REGISTRY:
        raise ValueError(f"No editor registered for '{model_id}'. Available: {list(_REGISTRY)}")
    instance = _REGISTRY[model_id](config, cost_tracker, concurrency)
    instance.model_id = model_id
    instance.log = get_logger(f"edit.{model_id}")
    return instance


def all_registered() -> list[str]:
    return list(_REGISTRY)


from . import (  # noqa: E402,F401
    flux_kontext,
    flux2_flex,
    bria_edit,
    firefly,
    photoroom,
    picsart,
    canva_leonardo,
)


def log_edit(result: EditResult) -> None:
    log_path = OUTPUTS_DIR / "metadata" / "edit_log.jsonl"
    append_jsonl(log_path, result.to_dict())
    if result.status in (EditStatus.FILTERED, EditStatus.ERROR):
        fail_path = OUTPUTS_DIR / "metadata" / "failures.jsonl"
        append_jsonl(fail_path, result.to_dict())
