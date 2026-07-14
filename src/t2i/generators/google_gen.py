"""Google Imagen / Nano Banana Pro via Gemini API.

Pattern: sync
  POST /v1beta/models/{model}:predict?key=... -> { predictions: [{ bytesBase64Encoded }] }
"""
from __future__ import annotations

from .base import BaseGenerator, _ContentFiltered, looks_like_filter, register


@register("nano_banana_pro")
class GoogleGenerator(BaseGenerator):
    provider = "google"

    async def _do_generate(self, prompt_text: str) -> tuple[bytes, dict]:
        url = f"{self.config['api_url']}:predict?key={self.api_key}"
        body = {
            "instances": [{"prompt": prompt_text}],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": "1:1",
            },
        }
        r = await self.client.post(url, json=body)
        if r.status_code == 400 and looks_like_filter(r.text):
            raise _ContentFiltered(r.text[:300], metadata={"status": 400})
        r.raise_for_status()
        data = r.json()
        preds = data.get("predictions", [])
        if not preds or "bytesBase64Encoded" not in preds[0]:
            # Check for safety-filtered response
            if data.get("predictions") == [] or "raiFilteredReason" in str(data):
                raise _ContentFiltered(str(data)[:300], metadata=data)
            raise RuntimeError(f"No image in Google response: {data}")
        return self._b64_to_bytes(preds[0]["bytesBase64Encoded"]), data
