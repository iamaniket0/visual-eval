"""Freepik Mystic 2.5 Fluid.

Pattern: async_poll
  POST /v1/ai/text-to-image (or /v1/ai/mystic) -> { data: { task_id, status } }
  GET  /v1/ai/mystic/{task_id}                 -> { data: { status, generated: [...] } }
"""
from __future__ import annotations

from .base import BaseGenerator, register


@register("freepik_mystic")
class FreepikGenerator(BaseGenerator):
    provider = "freepik"

    async def _do_generate(self, prompt_text: str) -> tuple[bytes, dict]:
        headers = {
            "x-freepik-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        body = {
            "prompt": prompt_text,
            "resolution": "2k",
            "aspect_ratio": "square_1_1",
            "model": "fluid",
            "engine": "magnific_sharpy",
        }
        r = await self.client.post(self.config["api_url"], headers=headers, json=body)
        r.raise_for_status()
        submit = r.json()
        task_id = submit.get("data", {}).get("task_id") or submit.get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id in Freepik response: {submit}")

        poll_url = f"{self.config['api_url']}/{task_id}"

        def ready(d: dict) -> bool:
            return d.get("data", {}).get("status") == "COMPLETED"

        def failed(d: dict) -> str | None:
            status = d.get("data", {}).get("status")
            if status in ("FAILED", "CANCELLED"):
                return f"{status}: {d.get('data', {}).get('error', '')}"
            return None

        final = await self._poll_until_ready(
            poll_url, headers=headers,
            poll_interval=self.config.get("poll_interval_sec", 3),
            max_wait=self.config.get("max_poll_wait_sec", 180),
            ready_check=ready, failed_check=failed,
        )
        urls = final.get("data", {}).get("generated", [])
        if not urls:
            raise RuntimeError(f"No generated images: {final}")
        img_bytes = await self._download(urls[0])
        return img_bytes, {"submit": submit, "final": final}
