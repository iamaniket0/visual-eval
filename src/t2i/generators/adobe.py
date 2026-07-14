"""Adobe Firefly Image Model 5.

Pattern: sync
  POST /v3/images/generate -> { outputs: [{ image: { presignedUrl | url } }] }

Note: requires IMS token exchange (client_id + client_secret) - we fetch it
on first use and cache it. Note that Firefly results are shared
only with Adobe per their customer agreement.
"""

from __future__ import annotations

import time

from src.core.utils import get_api_key

from .base import BaseGenerator, register


@register("adobe_firefly_5")
class AdobeGenerator(BaseGenerator):
    provider = "adobe"

    _token: str | None = None
    _token_expiry: float = 0.0

    async def _ensure_token(self) -> str:
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        client_secret = get_api_key("ADOBE_CLIENT_SECRET")
        if not client_secret:
            raise RuntimeError("ADOBE_CLIENT_SECRET not set")
        r = await self.client.post(
            "https://ims-na1.adobelogin.com/ims/token/v3",
            data={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": client_secret,
                "scope": "openid,AdobeID,firefly_api,ff_apis",
            },
        )
        r.raise_for_status()
        tok = r.json()
        self._token = tok["access_token"]
        self._token_expiry = time.time() + int(tok.get("expires_in", 3600))
        return self._token

    async def _do_generate(self, prompt_text: str) -> tuple[bytes, dict]:
        token = await self._ensure_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        body = {
            "prompt": prompt_text,
            "size": {"width": 1024, "height": 1024},
            "numVariations": 1,
        }
        r = await self.client.post(self.config["api_url"], headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        outputs = data.get("outputs", [])
        if not outputs:
            raise RuntimeError(f"No outputs in Firefly response: {data}")
        image_info = outputs[0].get("image", {})
        url = image_info.get("presignedUrl") or image_info.get("url")
        if not url:
            raise RuntimeError(f"No image URL: {data}")
        img_bytes = await self._download(url)
        return img_bytes, data
