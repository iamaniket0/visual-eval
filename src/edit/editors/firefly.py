"""Adobe Firefly Image Edit v3 — generative fill + instruction editing.

Pattern: sync
POST /v3/images/edit with source image + prompt → edited image
Requires Adobe IMS access token from client credentials flow.
"""

from __future__ import annotations

from src.core.utils import get_api_key

from .base import BaseEditor, register


@register("firefly")
class FireflyEditor(BaseEditor):
    provider = "adobe"

    async def _do_edit(
        self, source_image_path: str, instruction: str, mask_path: str | None = None
    ) -> tuple[bytes, dict]:
        access_token = await self._get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        body: dict = {
            "image": {
                "source": {
                    "uploadId": self._image_to_b64(source_image_path),
                }
            },
            "prompt": instruction,
        }
        if mask_path and self.supports_mask:
            body["image"]["mask"] = {
                "source": {
                    "uploadId": self._image_to_b64(mask_path),
                }
            }

        r = await self.client.post(self.config["api_url"], headers=headers, json=body)
        r.raise_for_status()
        data = r.json()

        outputs = data.get("outputs", [])
        if not outputs:
            raise RuntimeError(f"No outputs in Firefly response: {data}")

        image_data = outputs[0].get("image", {})
        if "url" in image_data:
            img_bytes = await self._download(image_data["url"])
        elif "base64" in image_data:
            img_bytes = self._b64_to_bytes(image_data["base64"])
        else:
            raise RuntimeError(f"No image data in Firefly output: {image_data}")

        return img_bytes, data

    async def _get_access_token(self) -> str:
        client_secret = get_api_key("ADOBE_CLIENT_SECRET")
        if not client_secret:
            raise RuntimeError("ADOBE_CLIENT_SECRET not set")
        r = await self.client.post(
            "https://ims-na1.adobelogin.com/ims/token/v3",
            data={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": client_secret,
                "scope": "openid,AdobeID,firefly_api",
            },
        )
        r.raise_for_status()
        return r.json()["access_token"]
