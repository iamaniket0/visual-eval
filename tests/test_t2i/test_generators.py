"""Tests for T2I generator base class (no API calls)."""

from src.core.utils import CostTracker
from src.t2i.generators import all_registered, get_generator
from src.t2i.generators.base import GenerationStatus, looks_like_filter


def test_all_models_registered():
    expected = {
        "flux2_max",
        "stable_image_ultra",
        "bria_fibo",
        "freepik_mystic",
        "xai_aurora",
        "nano_banana_pro",
        "gpt_image_15",
        "gpt_image_2",
        "canva_lucid_origin",
        "adobe_firefly_5",
        "midjourney_v8",
    }
    assert expected.issubset(set(all_registered()))


def test_filter_detection_positive():
    assert looks_like_filter("Your prompt violates our content policy.")
    assert looks_like_filter("safety system blocked this")
    assert looks_like_filter("Request blocked: NSFW content")


def test_filter_detection_negative():
    assert not looks_like_filter("Internal server error")
    assert not looks_like_filter("rate limit exceeded")
    assert not looks_like_filter("timeout")


def test_missing_api_key_yields_skipped(monkeypatch):
    import asyncio
    import tempfile
    from pathlib import Path

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    ct = CostTracker(hard_cap_usd=300)
    cfg = {
        "api_url": "https://example.invalid/v1",
        "api_key_env": "DEFINITELY_NOT_SET_ABCXYZ",
        "cost_per_image": 0.04,
        "pattern": "sync",
    }

    async def run():
        gen = get_generator("gpt_image_15", cfg, ct)
        async with gen:
            return await gen.generate("test_001", "a cat", Path(tempfile.gettempdir()))

    result = asyncio.run(run())
    assert result.status == GenerationStatus.SKIPPED
    assert "not set" in (result.error or "").lower()
