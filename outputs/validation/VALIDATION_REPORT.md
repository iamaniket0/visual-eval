# API Key Validation Report

**Date**: 2026-07-14
**Purpose**: Validate which API keys from the original t2i-eval and edit-eval projects are still active, and test end-to-end image generation.

## Key Status Summary

| Provider | Auth Valid | Can Generate | Notes |
|:---------|:-----------|:-------------|:------|
| **OpenRouter** | YES | YES | GPT Image works via proxy ($0.20/image). Can also reach Qwen judge models. |
| **Together AI** | YES | PARTIAL | Key authenticates. Qwen3.5-397B-A17B requires dedicated endpoint (non-serverless). Smaller models (Qwen3.5-9B) work. |
| **xAI** | YES | YES | Aurora generates images successfully. 6s latency. |
| **BFL** | YES | YES | FLUX 2 Max generates images. 28s async poll. Also covers FLUX Kontext + FLUX.2 Flex for editing. |
| **Bria** | YES | YES | FIBO generates images via async polling (202 → poll → COMPLETED). ~12s. |
| **Leonardo** (t2i) | YES | NO | Auth succeeds but only 150 subscription tokens remaining — not enough for generation. |
| **Leonardo** (edit) | YES | NO | Same situation — 150 tokens, insufficient for generation runs. |
| **Picsart** | YES | UNTESTED | Balance endpoint returns 200 — key authenticates. No edit test run (needs source image upload). |

## OpenRouter as Model Proxy

OpenRouter can substitute for missing direct API keys for these models:

| Model | OpenRouter ID | Covers Missing Key? |
|:------|:-------------|:-------------------|
| GPT Image 1.5 | `openai/gpt-5-image` | YES — replaces missing `OPENAI_API_KEY` |
| GPT Image 2 | `openai/gpt-5.4-image-2` | YES — replaces missing `OPENAI_API_KEY` |
| Qwen3.5-397B (judge) | `qwen/qwen3.5-397b-a17b` | YES — alternative to Together for judging |
| Google Imagen 3 | NOT AVAILABLE | NO |
| Stable Diffusion 3.5 | NOT AVAILABLE | NO |
| Freepik Mystic | NOT AVAILABLE | NO |
| Adobe Firefly | NOT AVAILABLE | NO |

**Bottom line**: OpenRouter covers GPT Image generation and Qwen judging, but NOT the diffusion model APIs (BFL, Stability, Bria, Freepik, Adobe, Leonardo). Those require direct provider keys.

## Generated Test Images

Two successful test generations saved in this directory:
- `xai_aurora_test.png` (145 KB) — "A red apple on a white plate"
- `flux2_max_test.png` (220 KB) — "A red apple on a white plate"

## Models Runnable Right Now

### T2I Generation (5 of 11 models)
- xai_aurora (direct xAI key)
- flux2_max (direct BFL key)
- bria_fibo (direct Bria key)
- gpt_image_15 (via OpenRouter)
- gpt_image_2 (via OpenRouter)

### Edit (4 of 7 models)
- flux_kontext (shared BFL key)
- flux2_flex (shared BFL key)
- bria_edit (shared Bria key)
- picsart (direct Picsart key)

### Judge
- Qwen3.5-397B-A17B via OpenRouter (Together requires dedicated endpoint)

### NOT Runnable (missing or exhausted keys)
- stable_image_ultra (no Stability key)
- freepik_mystic (no Freepik key)
- nano_banana_pro / Imagen 3 (no Google key)
- lucid_origin (Leonardo tokens exhausted)
- adobe_firefly_5 (no Adobe credentials)
- canva_leonardo (Leonardo tokens exhausted)
- photoroom (no Photoroom key)
- firefly edit (no Adobe credentials)
