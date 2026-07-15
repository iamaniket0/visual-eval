"""Base class for all image generators.

Supports two API patterns:
    1. sync        - POST returns the image (or URL/base64) directly
    2. async_poll  - POST returns a job id; GET polls until ready

All generators share:
    - Exponential backoff retry on 429/5xx
    - Content filter detection (logged as FILTERED, not retried)
    - Cost accounting via CostTracker
    - PNG normalization on save
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
from src.t2i import OUTPUTS_DIR


class GenerationStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FILTERED = "FILTERED"  # blocked by provider content policy
    ERROR = "ERROR"  # network/API failure after retries
    SKIPPED = "SKIPPED"  # missing API key or cap reached


@dataclass
class GenerationResult:
    prompt_id: str
    model: str
    status: GenerationStatus
    image_path: str | None = None
    cost_usd: float = 0.0
    duration_sec: float = 0.0
    error: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Seed index for multi-seed variance runs. Legacy records (no seed field)
    # and the first seed of any run are both treated as seed=0.
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# Phrases that indicate a content-policy block rather than a transient error.
# If any of these substrings appear in an error message, we log FILTERED and
# do NOT retry (retrying a modified prompt would corrupt the benchmark).
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


class BaseGenerator(ABC):
    """Abstract base. Concrete subclasses implement one provider."""

    model_id: str = ""  # our internal model key, e.g. "flux2_max"
    provider: str = ""

    def __init__(self, config: dict[str, Any], cost_tracker: CostTracker, concurrency: int = 4):
        self.config = config
        self.cost_tracker = cost_tracker
        self.log = get_logger(f"gen.{self.model_id}")
        self.api_key = get_api_key(config["api_key_env"])
        self.cost_per_image = float(config["cost_per_image"])
        self.pattern = config.get("pattern", "sync")
        self.semaphore = asyncio.Semaphore(concurrency)
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Generator must be used inside `async with`.")
        return self._client

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def generate(
        self, prompt_id: str, prompt_text: str, output_dir: Path, seed: int = 0
    ) -> GenerationResult:
        started = datetime.now(timezone.utc)

        if not self.api_key:
            return GenerationResult(
                prompt_id=prompt_id,
                model=self.model_id,
                status=GenerationStatus.SKIPPED,
                seed=seed,
                error=f"{self.config['api_key_env']} not set",
            )

        if not self.cost_tracker.check_cap():
            return GenerationResult(
                prompt_id=prompt_id,
                model=self.model_id,
                status=GenerationStatus.SKIPPED,
                seed=seed,
                error="Cost cap reached",
            )

        async with self.semaphore:
            try:
                image_bytes, raw_meta = await self._generate_with_retry(prompt_text, seed=seed)
            except _ContentFilteredError as e:
                return GenerationResult(
                    prompt_id=prompt_id,
                    model=self.model_id,
                    status=GenerationStatus.FILTERED,
                    seed=seed,
                    error=str(e),
                    raw_metadata=e.metadata,
                )
            except Exception as e:
                return GenerationResult(
                    prompt_id=prompt_id,
                    model=self.model_id,
                    status=GenerationStatus.ERROR,
                    seed=seed,
                    error=f"{type(e).__name__}: {e}",
                )

        # Filename scheme: seed 0 keeps the legacy path so existing runs stay
        # on disk; seeds >=1 get a "__s{n}.png" suffix.
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{prompt_id}.png" if seed == 0 else f"{prompt_id}__s{seed}.png"
        out_path = output_dir / fname
        self._save_png(image_bytes, out_path)
        self.cost_tracker.add(self.cost_per_image, model=self.model_id, stage="generation")

        duration = (datetime.now(timezone.utc) - started).total_seconds()
        return GenerationResult(
            prompt_id=prompt_id,
            model=self.model_id,
            status=GenerationStatus.SUCCESS,
            image_path=str(out_path),
            cost_usd=self.cost_per_image,
            duration_sec=round(duration, 2),
            raw_metadata=raw_meta,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------

    async def _generate_with_retry(
        self, prompt_text: str, seed: int = 0, max_retries: int = 3
    ) -> tuple[bytes, dict]:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await self._do_generate_with_seed(prompt_text, seed)
            except _ContentFilteredError:
                raise  # never retry content filter blocks
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                body_snippet = e.response.text[:500] if e.response is not None else ""
                if looks_like_filter(body_snippet):
                    raise _ContentFilteredError(
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
            except RuntimeError as e:
                # "Empty images in response" and similar adapter-level parse
                # failures that aren't HTTP-level retries but ARE transient -
                # observed on gpt-5-image via OpenRouter where the response
                # sometimes comes back HTTP 200 with choices[0].message.images
                # empty (soft refusal / load degradation). Retrying usually
                # succeeds. Narrowly scoped - we only retry RuntimeErrors
                # whose message starts with "No images" so we don't hide
                # other bugs.
                if attempt < max_retries - 1 and "no images" in str(e).lower():
                    delay = 2**attempt
                    self.log.warning("Empty images response, retrying in %ds", delay)
                    await asyncio.sleep(delay)
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
    async def _do_generate(self, prompt_text: str) -> tuple[bytes, dict]:
        """Return (image_bytes, raw_metadata). May raise _ContentFilteredError."""

    async def _do_generate_with_seed(self, prompt_text: str, seed: int) -> tuple[bytes, dict]:
        """Default seed-aware path: delegates to `_do_generate` and ignores
        the seed value. Appropriate for providers whose APIs don't accept an
        explicit seed (xAI, OpenAI DALL-E) - variance across "seeds" comes
        from repeat API calls and temperature/sampling stochasticity.

        Adapters whose APIs DO accept a seed parameter (BFL, Leonardo,
        Stability) should override this to pass `seed` into their request
        body when seed > 0. Seed == 0 keeps legacy behaviour so existing
        on-disk images don't need regenerating.
        """
        return await self._do_generate(prompt_text)

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _save_png(image_bytes: bytes, path: Path) -> None:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")  # type: ignore[assignment]
        img.save(path, format="PNG")

    @staticmethod
    def _b64_to_bytes(data: str) -> bytes:
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        return base64.b64decode(data)

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
        """Generic poll helper for async_poll providers.

        ready_check(response_json) -> bool  : True when job is done.
        failed_check(response_json) -> str|None : reason if job failed else None.
        """
        elapsed = 0.0
        while elapsed < max_wait:
            r = await self.client.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
            if failed_check and (reason := failed_check(data)):
                if looks_like_filter(reason):
                    raise _ContentFilteredError(reason, metadata=data)
                raise RuntimeError(f"Job failed: {reason}")
            if ready_check(data):
                return data  # type: ignore[no-any-return]
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        raise TimeoutError(f"Job did not complete within {max_wait}s")


class _ContentFilteredError(Exception):
    def __init__(self, msg: str, metadata: dict | None = None):
        super().__init__(msg)
        self.metadata = metadata or {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseGenerator]] = {}


def register(model_id: str):
    """Register a generator class under `model_id`.

    A class may be registered under multiple ids (e.g. one OpenAIGenerator
    class serves both `gpt_image_15` and `gpt_image_2`). The class's own
    `model_id` class attribute is set to the last registration for logging
    fallback, but `get_generator` always injects the looked-up key onto the
    instance after construction so each instance reports its correct name.
    """

    def deco(cls):
        cls.model_id = model_id
        _REGISTRY[model_id] = cls
        return cls

    return deco


def get_generator(
    model_id: str, config: dict[str, Any], cost_tracker: CostTracker, concurrency: int = 4
) -> BaseGenerator:
    if model_id not in _REGISTRY:
        raise ValueError(f"No generator registered for '{model_id}'. Available: {list(_REGISTRY)}")
    instance = _REGISTRY[model_id](config, cost_tracker, concurrency)
    # Override so this instance reports the exact key it was looked up with,
    # even when the underlying class is shared across multiple model_ids.
    # `self.model_id` drives logger name, GenerationResult.model, and cost
    # tracking; they all need to reflect the specific model, not the class
    # alias. Also refresh the logger because it was created with the stale
    # class-attribute name during __init__.
    instance.model_id = model_id
    instance.log = get_logger(f"gen.{model_id}")
    return instance


def all_registered() -> list[str]:
    return list(_REGISTRY)


# Import concrete generators so they register
from . import (  # noqa: E402,F401
    adobe,
    bria,
    flux,
    freepik,
    google_gen,
    leonardo,
    midjourney,
    openai_gen,
    stability,
    xai,
)


def log_generation(result: GenerationResult) -> None:
    """Append a generation log line."""
    log_path = OUTPUTS_DIR / "metadata" / "generation_log.jsonl"
    append_jsonl(log_path, result.to_dict())
    if result.status in (GenerationStatus.FILTERED, GenerationStatus.ERROR):
        fail_path = OUTPUTS_DIR / "metadata" / "failures.jsonl"
        append_jsonl(fail_path, result.to_dict())
