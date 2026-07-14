"""Stable Image Ultra via Stability AI.

Pattern: sync (multipart/form-data)
  POST /v2beta/stable-image/generate/ultra -> image bytes directly
"""
from __future__ import annotations

from .base import BaseGenerator, _ContentFiltered, looks_like_filter, register


@register("stable_image_ultra")
class StabilityGenerator(BaseGenerator):
    provider = "stability"

    async def _do_generate(self, prompt_text: str) -> tuple[bytes, dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "image/*",
        }
        # Multipart/form-data with empty "none" field per Stability docs
        files = {"none": (None, "")}
        data = {
            "prompt": prompt_text,
            "output_format": "png",
            "aspect_ratio": "1:1",
        }
        r = await self.client.post(
            self.config["api_url"], headers=headers, data=data, files=files,
        )
        if r.status_code == 403 and looks_like_filter(r.text):
            raise _ContentFiltered(r.text[:300], metadata={"status": 403})
        r.raise_for_status()
        return r.content, {
            "content_type": r.headers.get("content-type"),
            "finish_reason": r.headers.get("finish-reason"),
            "seed": r.headers.get("seed"),
        }
