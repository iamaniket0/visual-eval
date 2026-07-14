"""Canva/Leonardo.ai — image-to-image editing via Leonardo API.

Pattern: async_poll
1. POST /api/rest/v1/init-image → pre-signed S3 upload URL + fields
2. POST to S3 with pre-signed fields + image file
3. POST /api/rest/v1/generations with init_image_id + prompt → generation_id
4. GET  /api/rest/v1/generations/{id} until status == "COMPLETE" → image URL
"""
from __future__ import annotations

import asyncio
import json as json_mod

from .base import BaseEditor, register


@register("canva_leonardo")
class CanvaLeonardoEditor(BaseEditor):
    provider = "leonardo"

    async def _do_edit(self, source_image_path: str, instruction: str,
                       mask_path: str | None = None) -> tuple[bytes, dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        ext = source_image_path.rsplit(".", 1)[-1].lower()
        if ext not in ("jpg", "jpeg", "png", "webp"):
            ext = "jpg"

        upload_resp = await self.client.post(
            "https://cloud.leonardo.ai/api/rest/v1/init-image",
            headers=headers,
            json={"extension": ext},
        )
        upload_resp.raise_for_status()
        upload_data = upload_resp.json()

        init_fields = upload_data.get("uploadInitImage", {})
        init_image_id = init_fields.get("id")
        upload_url = init_fields.get("url")
        fields_raw = init_fields.get("fields", "{}")
        fields = json_mod.loads(fields_raw) if isinstance(fields_raw, str) else (fields_raw or {})

        with open(source_image_path, "rb") as f:
            img_bytes_raw = f.read()

        key = fields.pop("key", f"{init_image_id}.{ext}")
        form: dict[str, str] = {"key": key}
        form.update(fields)

        resp = await self.client.post(
            upload_url,
            data=form,
            files={"file": (f"image.{ext}", img_bytes_raw, f"image/{ext}")},
        )
        if resp.status_code >= 400:
            self.log.warning("S3 upload returned %d: %s", resp.status_code, resp.text[:300])

        model_id = self.config.get("leonardo_model_id", "de7d3faf-762f-48e0-b3b7-9d0ac3a3fcf3")

        body: dict = {
            "prompt": instruction,
            "modelId": model_id,
            "init_image_id": init_image_id,
            "num_images": 1,
            "width": 1024,
            "height": 1024,
            "init_strength": 0.3,
        }

        r = await self.client.post(self.config["api_url"], headers=headers, json=body)
        r.raise_for_status()
        data = r.json()

        gen_id = data.get("sdGenerationJob", {}).get("generationId")
        if not gen_id:
            raise RuntimeError(f"No generationId in Leonardo response: {data}")

        poll_interval = self.config.get("poll_interval_sec", 3)
        max_wait = self.config.get("max_poll_wait_sec", 180)
        elapsed = 0.0

        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            pr = await self.client.get(
                f"https://cloud.leonardo.ai/api/rest/v1/generations/{gen_id}",
                headers=headers,
            )
            pr.raise_for_status()
            result = pr.json()
            status = result.get("generations_by_pk", {}).get("status", "")

            if status == "COMPLETE":
                images = result.get("generations_by_pk", {}).get("generated_images", [])
                if not images:
                    raise RuntimeError(f"No images in completed Leonardo response: {result}")
                img_url = images[0].get("url")
                if not img_url:
                    raise RuntimeError(f"No URL in Leonardo image: {images[0]}")
                img_bytes = await self._download(img_url)
                return img_bytes, result

            if status == "FAILED":
                raise RuntimeError(f"Leonardo edit failed: {result}")

        raise TimeoutError(f"Leonardo poll timed out after {max_wait}s")
