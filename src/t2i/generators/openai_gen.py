"""GPT Image 1.5 — dual-path generator.

Routing is controlled by settings.api_routing.gpt_image:

  openai      -> direct POST /v1/images/generations
                 body: {model, prompt, n, size}
                 response: data[0].b64_json | data[0].url

  openrouter  -> POST /v1/chat/completions on OpenRouter
                 body: {model, modalities:[image,text], messages:[{role,content}]}
                 response images live in choices[0].message.images as data URLs
                 (or dicts of shape {type: image_url, image_url: {url: data:...}})

Both paths return the same (image_bytes, raw_metadata) tuple so the
BaseGenerator contract is unchanged. See config/models.yaml for the
openrouter_model and human-readable labels.

Caveat for methodology: OpenRouter's `openai/gpt-5-image` is a newer
generation than OpenAI's direct `gpt-image-1` / "GPT Image 1.5". Which
endpoint was actually queried is captured in raw_metadata and surfaced
via `config.model_label_{direct,openrouter}` so the report can label it
honestly.
"""
from __future__ import annotations

from typing import Any

from src.core.utils import get_api_key
from src.t2i import load_settings
from .base import BaseGenerator, _ContentFiltered, looks_like_filter, register


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://visual-eval-benchmark",
    "X-Title": "visual-eval-benchmark",
}


def _extract_openrouter_image_source(data: dict) -> str:
    """Pull the first image URL (or data URL) out of an OpenRouter chat
    completion response. Raises RuntimeError if nothing usable is found.

    Accepts the three shapes we've seen in the wild:
      - choices[0].message.images = ["data:image/png;base64,..."]
      - choices[0].message.images = [{"type": "image_url", "image_url": {"url": "..."}}]
      - choices[0].message.images = [{"image_url": "...", "b64_json": "..."}]
    """
    try:
        images = data["choices"][0]["message"]["images"]
    except (KeyError, IndexError, TypeError):
        images = None
    if not images:
        raise RuntimeError(f"No images in OpenRouter response: {str(data)[:300]}")

    first = images[0]
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        iu = first.get("image_url")
        if isinstance(iu, dict):
            url = iu.get("url")
            if url:
                return url
        elif isinstance(iu, str):
            return iu
        url = first.get("url")
        if url:
            return url
        b64 = first.get("b64_json")
        if b64:
            return b64 if b64.startswith("data:") else f"data:image/png;base64,{b64}"
    raise RuntimeError(f"Unrecognized image shape in OpenRouter response: {str(data)[:300]}")


@register("gpt_image_15")
@register("gpt_image_2")
class OpenAIGenerator(BaseGenerator):
    provider = "openai"

    def __init__(self, config: dict[str, Any], cost_tracker, concurrency: int = 4):
        super().__init__(config, cost_tracker, concurrency)
        settings = load_settings()
        self.routing = settings.get("api_routing", {}).get("gpt_image", "openai")

        if self.routing == "openrouter":
            # Swap auth + endpoint to OpenRouter. Clone config so the upstream
            # dict (shared across generators) isn't mutated.
            self.api_key = get_api_key("OPENROUTER_API_KEY")
            self.config = {**config, "api_key_env": "OPENROUTER_API_KEY"}
            self._endpoint = OPENROUTER_ENDPOINT
            self._model = config.get("openrouter_model", "openai/gpt-5-image")
        else:
            self._endpoint = config["api_url"]
            self._model = config.get("model_name", "gpt-image-1")

    async def _do_generate(self, prompt_text: str) -> tuple[bytes, dict]:
        if self.routing == "openrouter":
            return await self._generate_openrouter(prompt_text)
        return await self._generate_direct(prompt_text)

    async def _generate_direct(self, prompt_text: str) -> tuple[bytes, dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "prompt": prompt_text,
            "n": 1,
            "size": "1024x1024",
        }
        r = await self.client.post(self._endpoint, headers=headers, json=body)
        if r.status_code == 400 and looks_like_filter(r.text):
            raise _ContentFiltered(r.text[:300], metadata={"status": 400})
        r.raise_for_status()
        data = r.json()
        item = data.get("data", [{}])[0]
        if "b64_json" in item:
            return self._b64_to_bytes(item["b64_json"]), data
        if "url" in item:
            return await self._download(item["url"]), data
        raise RuntimeError(f"No image in OpenAI response: {data}")

    async def _generate_openrouter(self, prompt_text: str) -> tuple[bytes, dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **OPENROUTER_HEADERS,
        }
        body = {
            "model": self._model,
            "modalities": ["image", "text"],
            "messages": [{"role": "user", "content": prompt_text}],
        }
        r = await self.client.post(self._endpoint, headers=headers, json=body)
        if r.status_code == 400 and looks_like_filter(r.text):
            raise _ContentFiltered(r.text[:300], metadata={"status": 400})
        r.raise_for_status()
        data = r.json()
        url = _extract_openrouter_image_source(data)
        if url.startswith("data:"):
            return self._b64_to_bytes(url), data
        return await self._download(url), data
