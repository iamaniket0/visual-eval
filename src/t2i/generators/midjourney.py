"""Midjourney V8 via PiAPI (unofficial wrapper).

OPTIONAL - Flagged as ToS-risky. If PiAPI key is missing,
the generator will SKIP cleanly rather than error.

Pattern: async_poll
  POST /api/v1/task        -> { data: { task_id } }
  GET  /api/v1/task/{id}   -> { data: { status, output: { image_url(s) } } }
"""

from __future__ import annotations

from .base import BaseGenerator, register


@register("midjourney_v8")
class MidjourneyGenerator(BaseGenerator):
    provider = "piapi"

    async def _do_generate(self, prompt_text: str) -> tuple[bytes, dict]:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        body = {
            "model": "midjourney",
            "task_type": "imagine",
            "input": {
                "prompt": prompt_text,
                "aspect_ratio": "1:1",
                "process_mode": "fast",
            },
        }
        r = await self.client.post(self.config["api_url"], headers=headers, json=body)
        r.raise_for_status()
        submit = r.json()
        task_id = submit.get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id in PiAPI response: {submit}")

        poll_url = f"{self.config['api_url']}/{task_id}"

        def ready(d: dict) -> bool:
            return d.get("data", {}).get("status") == "completed"

        def failed(d: dict) -> str | None:
            status = d.get("data", {}).get("status")
            if status in ("failed", "cancelled"):
                err = d.get("data", {}).get("error", {})
                return f"{status}: {err}"
            return None

        final = await self._poll_until_ready(
            poll_url,
            headers=headers,
            poll_interval=self.config.get("poll_interval_sec", 5),
            max_wait=self.config.get("max_poll_wait_sec", 300),
            ready_check=ready,
            failed_check=failed,
        )
        output = final.get("data", {}).get("output", {})
        url = output.get("image_url") or (output.get("image_urls") or [None])[0]
        if not url:
            raise RuntimeError(f"No image_url in final: {final}")
        img_bytes = await self._download(url)
        return img_bytes, {"submit": submit, "final": final}
