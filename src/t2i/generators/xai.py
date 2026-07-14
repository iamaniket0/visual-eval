"""xAI Grok Imagine (Aurora MoE).

Pattern: sync (OpenAI-compatible images endpoint)
  POST /v1/images/generations -> { data: [{ b64_json | url }] }

Multi-seed variance note: the xAI API does not accept a `seed` parameter.
Repeat calls with identical input return different images due to the API's
internal sampling stochasticity, so this adapter inherits the default
`_do_generate_with_seed` behaviour (ignore seed value, call the endpoint).
Run the full seed loop via `scripts.run_generation --seeds N`; the outputs
will differ across seeds even though no seed value is explicitly sent.
"""
from __future__ import annotations

from .base import BaseGenerator, register


@register("xai_aurora")
class XAIGenerator(BaseGenerator):
    provider = "xai"

    async def _do_generate(self, prompt_text: str) -> tuple[bytes, dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.config.get("model_name", "grok-imagine-pro"),
            "prompt": prompt_text,
            "n": 1,
            "response_format": "b64_json",
        }
        r = await self.client.post(self.config["api_url"], headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        item = data.get("data", [{}])[0]
        if "b64_json" in item:
            return self._b64_to_bytes(item["b64_json"]), data
        if "url" in item:
            return await self._download(item["url"]), data
        raise RuntimeError(f"No image in xAI response: {data}")
