"""Picsart Editing API — AI-powered instruction editing.

Pattern: sync
POST /tools/1.0/edit with source image + prompt → edited image URL
"""
from __future__ import annotations

from .base import BaseEditor, register


@register("picsart")
class PicsartEditor(BaseEditor):
    provider = "picsart"

    async def _do_edit(self, source_image_path: str, instruction: str,
                       mask_path: str | None = None) -> tuple[bytes, dict]:
        headers = {
            "X-Picsart-API-Key": self.api_key,
        }

        with open(source_image_path, "rb") as f:
            source_bytes = f.read()

        files: dict = {
            "image": ("source.png", source_bytes, "image/png"),
        }
        data: dict = {
            "prompt": instruction,
        }
        if mask_path and self.supports_mask:
            with open(mask_path, "rb") as mf:
                mask_bytes = mf.read()
            files["mask"] = ("mask.png", mask_bytes, "image/png")

        r = await self.client.post(
            self.config["api_url"],
            headers=headers,
            data=data,
            files=files,
        )
        r.raise_for_status()
        resp_data = r.json()

        image_url = resp_data.get("data", {}).get("url")
        if not image_url:
            raise RuntimeError(f"No image URL in Picsart response: {resp_data}")

        img_bytes = await self._download(image_url)
        return img_bytes, resp_data
