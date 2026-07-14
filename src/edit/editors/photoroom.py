"""Photoroom Edit API — background/foreground manipulation.

Pattern: sync
POST /v1/edit with multipart form data (source image + prompt) → edited image bytes
"""

from __future__ import annotations

from .base import BaseEditor, register


@register("photoroom")
class PhotoroomEditor(BaseEditor):
    provider = "photoroom"

    async def _do_edit(
        self, source_image_path: str, instruction: str, mask_path: str | None = None
    ) -> tuple[bytes, dict]:
        headers = {
            "x-api-key": self.api_key,
        }

        with open(source_image_path, "rb") as f:
            source_bytes = f.read()

        files = {
            "imageFile": ("source.png", source_bytes, "image/png"),
        }
        data = {
            "prompt": instruction,
        }

        r = await self.client.post(
            self.config["api_url"],
            headers=headers,
            data=data,
            files=files,
        )
        r.raise_for_status()

        content_type = r.headers.get("content-type", "")
        if "json" in content_type:
            resp_data = r.json()
            if "result_url" in resp_data:
                img_bytes = await self._download(resp_data["result_url"])
                return img_bytes, resp_data
            if "image_url" in resp_data:
                img_bytes = await self._download(resp_data["image_url"])
                return img_bytes, resp_data
            raise RuntimeError(f"Unexpected JSON from Photoroom: {resp_data}")

        return r.content, {"content_type": content_type}
