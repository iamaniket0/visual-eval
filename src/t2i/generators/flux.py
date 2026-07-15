"""FLUX.2 [max] via Black Forest Labs API.

Pattern: async_poll
  POST /v1/flux-pro-1.1  -> { id, polling_url }
  GET  polling_url        -> { status, result: { sample } }
"""

from __future__ import annotations

from typing import Any

from .base import BaseGenerator, register


@register("flux2_max")
class FluxGenerator(BaseGenerator):
    provider = "bfl"

    async def _do_generate(self, prompt_text: str) -> tuple[bytes, dict]:
        assert self.api_key is not None
        headers = {"x-key": self.api_key, "Content-Type": "application/json"}
        body: dict[str, Any] = {
            "prompt": prompt_text,
            "width": 1024,
            "height": 1024,
            "output_format": "png",
            "safety_tolerance": 2,
        }
        r = await self.client.post(self.config["api_url"], headers=headers, json=body)
        r.raise_for_status()
        submit = r.json()
        polling_url = submit.get("polling_url") or submit.get("id")
        if not polling_url:
            raise RuntimeError(f"No polling_url in response: {submit}")

        def ready(d: dict) -> bool:
            return d.get("status") in ("Ready", "Complete")

        def failed(d: dict) -> str | None:
            s = d.get("status", "")
            if s in ("Failed", "Error", "Content Moderated", "Request Moderated"):
                return s + ": " + str(d.get("details", ""))  # type: ignore[no-any-return]
            return None

        final = await self._poll_until_ready(
            polling_url,
            headers=headers,
            poll_interval=self.config.get("poll_interval_sec", 2),
            max_wait=self.config.get("max_poll_wait_sec", 120),
            ready_check=ready,
            failed_check=failed,
        )
        result = final.get("result", {})
        sample = result.get("sample")
        if not sample:
            raise RuntimeError(f"No sample in final response: {final}")
        img_bytes = (
            await self._download(sample)
            if sample.startswith("http")
            else self._b64_to_bytes(sample)
        )
        return img_bytes, {"submit": submit, "final": final}
