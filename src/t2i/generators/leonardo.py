"""Canva Lucid Origin via Leonardo.ai.

Pattern: async_poll
  POST /api/rest/v1/generations         -> { sdGenerationJob: { generationId } }
  GET  /api/rest/v1/generations/{id}    -> { generations_by_pk: { status, generated_images } }
"""
from __future__ import annotations

from .base import BaseGenerator, register


@register("canva_lucid_origin")
class LeonardoGenerator(BaseGenerator):
    provider = "leonardo"

    async def _do_generate(self, prompt_text: str) -> tuple[bytes, dict]:
        # Stateless default path (seed=0). Production callers route through
        # `_do_generate_with_seed` below so multi-seed runs get deterministic
        # variation via the provider's seed parameter.
        return await self._do_generate_with_seed(prompt_text, seed=0)

    async def _do_generate_with_seed(self, prompt_text: str,
                                      seed: int) -> tuple[bytes, dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        body = {
            "modelId": self.config["model_id"],
            "prompt": prompt_text,
            "width": 1024,
            "height": 1024,
            "num_images": 1,
        }
        # Leonardo accepts an explicit seed; pass one only when caller asked
        # for a specific variance trial (seed > 0). Seed 0 keeps provider
        # default behaviour so earlier on-disk generations stay reproducible.
        if seed > 0:
            body["seed"] = seed
        r = await self.client.post(self.config["api_url"], headers=headers, json=body)
        r.raise_for_status()
        submit = r.json()
        gen_id = submit.get("sdGenerationJob", {}).get("generationId")
        if not gen_id:
            raise RuntimeError(f"No generationId: {submit}")

        poll_url = f"{self.config['api_url']}/{gen_id}"

        def ready(d: dict) -> bool:
            return d.get("generations_by_pk", {}).get("status") == "COMPLETE"

        def failed(d: dict) -> str | None:
            status = d.get("generations_by_pk", {}).get("status")
            if status == "FAILED":
                return "FAILED"
            return None

        final = await self._poll_until_ready(
            poll_url, headers=headers,
            poll_interval=self.config.get("poll_interval_sec", 3),
            max_wait=self.config.get("max_poll_wait_sec", 180),
            ready_check=ready, failed_check=failed,
        )
        imgs = final.get("generations_by_pk", {}).get("generated_images", [])
        if not imgs:
            raise RuntimeError(f"No images in final: {final}")
        img_bytes = await self._download(imgs[0]["url"])
        return img_bytes, {"submit": submit, "final": final}
