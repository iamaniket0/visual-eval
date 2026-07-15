"""Bria FIBO Edit — commercial-safe image editing.

Pattern: async_poll
POST /v2/image/edit with source image + edit prompt → request_id + status_url
Poll status_url until status == "completed" → result URL
"""

from __future__ import annotations

import asyncio

from .base import BaseEditor, register


@register("bria_edit")
class BriaEditEditor(BaseEditor):
    provider = "bria"

    async def _do_edit(
        self, source_image_path: str, instruction: str, mask_path: str | None = None
    ) -> tuple[bytes, dict]:
        assert self.api_key is not None
        headers = {
            "api_token": self.api_key,
            "Content-Type": "application/json",
        }
        body: dict = {
            "images": [self._image_to_b64(source_image_path)],
            "instruction": instruction,
        }
        if mask_path and self.supports_mask:
            body["mask"] = self._image_to_b64(mask_path)

        r = await self.client.post(self.config["api_url"], headers=headers, json=body)
        r.raise_for_status()
        data = r.json()

        status_url = data.get("status_url")
        if not status_url:
            raise RuntimeError(f"No status_url in Bria edit response: {data}")

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
                    url = (
                        result.get("image_url")
                        or result.get("url")
                        or (result.get("urls") or [None])[0]
                    )
                elif isinstance(result, str):
                    url = result
                if not url:
                    raise RuntimeError(f"No image URL in Bria completed response: {status_data}")
                img_bytes = await self._download(url)
                return img_bytes, status_data

            if status in ("FAILED", "ERROR"):
                raise RuntimeError(f"Bria edit failed: {status_data}")

        raise TimeoutError(f"Bria edit poll timed out after {max_wait}s")
