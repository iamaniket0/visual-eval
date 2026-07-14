"""FLUX Kontext Pro — BFL's instruction-based image editor.

Pattern: async_poll
POST /v1/flux-kontext-pro with source image (base64) + prompt → job id
GET  /v1/get_result?id=... until status == "Ready" → image URL
"""

from __future__ import annotations

import asyncio

from .base import BaseEditor, register


@register("flux_kontext")
class FluxKontextEditor(BaseEditor):
    provider = "bfl"

    async def _do_edit(
        self, source_image_path: str, instruction: str, mask_path: str | None = None
    ) -> tuple[bytes, dict]:
        headers = {
            "X-Key": self.api_key,
            "Content-Type": "application/json",
        }

        body: dict = {
            "prompt": instruction,
            "input_image": self._image_to_b64(source_image_path),
        }
        if mask_path and self.supports_mask:
            body["mask"] = self._image_to_b64(mask_path)

        r = await self.client.post(self.config["api_url"], headers=headers, json=body)
        r.raise_for_status()
        data = r.json()

        job_id = data.get("id")
        if not job_id:
            raise RuntimeError(f"No job id in Flux Kontext response: {data}")

        poll_url = data.get("polling_url") or f"https://api.bfl.ai/v1/get_result?id={job_id}"
        poll_interval = self.config.get("poll_interval_sec", 2)
        max_wait = self.config.get("max_poll_wait_sec", 120)
        elapsed = 0.0

        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            pr = await self.client.get(poll_url, headers={"X-Key": self.api_key})
            pr.raise_for_status()
            result = pr.json()
            status = result.get("status", "").lower()

            if status == "ready":
                img_url = result.get("result", {}).get("sample")
                if not img_url:
                    raise RuntimeError(f"No image URL in ready response: {result}")
                img_bytes = await self._download(img_url)
                return img_bytes, result

            if status in ("failed", "error"):
                raise RuntimeError(f"Flux Kontext edit failed: {result}")

        raise TimeoutError(f"Flux Kontext poll timed out after {max_wait}s")
