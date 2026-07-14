"""Bria FIBO text-to-image.

Supports both v1 (legacy sync) and v2 (current async) API.
v2 pattern: POST /v2/image/generate -> { request_id, status_url }
            poll status_url until status == "completed" -> { result: [{ url }] }
v1 fallback: POST /v1/text-to-image -> { result: [{ urls: [...] }] }

The adapter auto-detects which version to use based on `api_url` in config.
"""
from __future__ import annotations

import asyncio

from .base import BaseGenerator, register


@register("bria_fibo")
class BriaGenerator(BaseGenerator):
    provider = "bria"

    async def _do_generate(self, prompt_text: str) -> tuple[bytes, dict]:
        api_url = self.config["api_url"]
        if "/v2/" in api_url:
            return await self._do_generate_v2(prompt_text)
        return await self._do_generate_v1(prompt_text)

    async def _do_generate_v2(self, prompt_text: str) -> tuple[bytes, dict]:
        headers = {"api_token": self.api_key, "Content-Type": "application/json"}
        body = {
            "prompt": prompt_text,
            "num_results": 1,
            "aspect_ratio": "1:1",
        }
        r = await self.client.post(self.config["api_url"], headers=headers, json=body)
        r.raise_for_status()
        data = r.json()

        status_url = data.get("status_url")
        if not status_url:
            raise RuntimeError(f"No status_url in Bria v2 response: {data}")

        poll_interval = self.config.get("poll_interval_sec", 3)
        max_wait = self.config.get("max_poll_wait_sec", 180)
        elapsed = 0.0

        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            sr = await self.client.get(status_url, headers={"api_token": self.api_key})
            sr.raise_for_status()
            status_data = sr.json()
            status = status_data.get("status", "").upper()

            if status == "COMPLETED":
                result = status_data.get("result", {})
                if isinstance(result, list):
                    result = result[0] if result else {}
                url = None
                if isinstance(result, dict):
                    url = (result.get("image_url")
                           or result.get("url")
                           or (result.get("urls") or [None])[0])
                elif isinstance(result, str):
                    url = result
                if not url:
                    raise RuntimeError(f"No image URL in Bria v2 completed response: {status_data}")
                img_bytes = await self._download(url)
                return img_bytes, status_data

            if status in ("FAILED", "ERROR"):
                raise RuntimeError(f"Bria v2 generation failed: {status_data}")

        raise RuntimeError(f"Bria v2 poll timed out after {max_wait}s")

    async def _do_generate_v1(self, prompt_text: str) -> tuple[bytes, dict]:
        headers = {"api_token": self.api_key, "Content-Type": "application/json"}
        body = {
            "prompt": prompt_text,
            "num_results": 1,
            "aspect_ratio": "1:1",
            "sync": True,
        }
        r = await self.client.post(self.config["api_url"], headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        result = data.get("result", [])
        url = None
        if result:
            first = result[0]
            if isinstance(first, dict):
                url = (first.get("urls") or [None])[0] or first.get("url")
            elif isinstance(first, list) and first:
                url = first[0]
        if not url:
            raise RuntimeError(f"No image URL in Bria response: {data}")
        img_bytes = await self._download(url)
        return img_bytes, data
