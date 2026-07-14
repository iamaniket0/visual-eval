"""Download 165 license-free source images from Unsplash for the benchmark.

Usage:
    python -m scripts.download_source_images
    python -m scripts.download_source_images --api pexels
    python -m scripts.download_source_images --dry-run

Requires either UNSPLASH_ACCESS_KEY or PEXELS_API_KEY in .env.

Categories:
  - instruction_boundary (55): people wearing clothes, objects on backgrounds
  - multi_turn (55): portraits with accessories, scenes with removable elements
  - fine_detail (55): products with text labels, textured objects, fine edges
"""

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from src.edit.prompt_loader import load_all_prompts
from src.core.utils import get_api_key, get_logger
from src.edit import PROMPTS_DIR

log = get_logger("download_source_images")

SEARCH_QUERIES = {
    "instruction_boundary": [
        "person wearing shirt portrait",
        "car parked street",
        "room interior furniture",
        "person standing outdoors",
        "house with garden",
    ],
    "multi_turn": [
        "portrait person accessories sunglasses",
        "living room with furniture",
        "person street fashion",
        "park bench trees",
        "kitchen table food",
    ],
    "fine_detail": [
        "product label closeup",
        "textured fabric knit",
        "jewelry ring detail",
        "handwritten note paper",
        "flower petal macro",
    ],
}


async def download_unsplash(
    prompt_id: str, query: str, output_path: Path, client: httpx.AsyncClient, api_key: str
) -> bool:
    """Download a single image from Unsplash."""
    try:
        resp = await client.get(
            "https://api.unsplash.com/photos/random",
            params={"query": query, "orientation": "squarish"},
            headers={"Authorization": f"Client-ID {api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()
        img_url = data["urls"]["regular"]

        img_resp = await client.get(img_url)
        img_resp.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(img_resp.content)

        log.info("Downloaded %s → %s", prompt_id, output_path.name)
        return True
    except Exception as e:
        log.warning("Failed to download %s: %s", prompt_id, e)
        return False


async def download_pexels(
    prompt_id: str,
    query: str,
    output_path: Path,
    client: httpx.AsyncClient,
    api_key: str,
    page: int = 1,
) -> bool:
    """Download a single image from Pexels."""
    try:
        resp = await client.get(
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": 1, "page": page},
            headers={"Authorization": api_key},
        )
        resp.raise_for_status()
        data = resp.json()
        photos = data.get("photos", [])
        if not photos:
            log.warning("No Pexels results for %s query='%s'", prompt_id, query)
            return False

        img_url = photos[0]["src"]["large"]
        img_resp = await client.get(img_url)
        img_resp.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(img_resp.content)

        log.info("Downloaded %s → %s", prompt_id, output_path.name)
        return True
    except Exception as e:
        log.warning("Failed to download %s: %s", prompt_id, e)
        return False


async def download_placeholder(prompt_id: str, output_path: Path) -> bool:
    """Create a placeholder image when no API key is available."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (1024, 1024), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        text = f"SOURCE\n{prompt_id}"
        draw.text((512, 512), text, fill=(100, 100, 100), anchor="mm")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="JPEG", quality=90)
        log.info("Created placeholder for %s", prompt_id)
        return True
    except Exception as e:
        log.warning("Failed to create placeholder for %s: %s", prompt_id, e)
        return False


async def main_async(args):
    prompts = load_all_prompts()
    source_dir = PROMPTS_DIR / "source_images"
    source_dir.mkdir(parents=True, exist_ok=True)

    to_download = []
    for p in prompts:
        src = p.get("source_image", "")
        if not src:
            continue
        out_path = PROMPTS_DIR / src
        if out_path.exists() and not args.force:
            continue
        to_download.append((p["prompt_id"], p.get("sub_category", ""), out_path))

    if not to_download:
        log.info("All %d source images already exist", len(prompts))
        return

    if args.dry_run:
        print(f"Would download {len(to_download)} images")
        return

    unsplash_key = get_api_key("UNSPLASH_ACCESS_KEY")
    pexels_key = get_api_key("PEXELS_API_KEY")

    use_api = args.api
    if use_api == "unsplash" and not unsplash_key:
        log.warning("UNSPLASH_ACCESS_KEY not set, falling back to placeholders")
        use_api = "placeholder"
    elif use_api == "pexels" and not pexels_key:
        log.warning("PEXELS_API_KEY not set, falling back to placeholders")
        use_api = "placeholder"
    elif not unsplash_key and not pexels_key:
        log.warning("No image API key set, creating placeholder images")
        use_api = "placeholder"
    elif use_api == "auto":
        use_api = "unsplash" if unsplash_key else "pexels"

    log.info("Downloading %d images via %s", len(to_download), use_api)

    sem = asyncio.Semaphore(4)
    success = 0

    async with httpx.AsyncClient(timeout=30.0) as client:

        async def _download(pid, subcat, out_path, idx):
            nonlocal success
            queries = SEARCH_QUERIES.get(subcat, SEARCH_QUERIES["instruction_boundary"])
            query = queries[idx % len(queries)]

            async with sem:
                if use_api == "unsplash":
                    ok = await download_unsplash(pid, query, out_path, client, unsplash_key)
                elif use_api == "pexels":
                    ok = await download_pexels(
                        pid, query, out_path, client, pexels_key, page=idx + 1
                    )
                else:
                    ok = await download_placeholder(pid, out_path)
                if ok:
                    success += 1

        tasks = [_download(pid, subcat, out, i) for i, (pid, subcat, out) in enumerate(to_download)]
        await asyncio.gather(*tasks)

    print(f"\nDownloaded {success}/{len(to_download)} source images")


def main():
    ap = argparse.ArgumentParser(description="Download source images for edit benchmark")
    ap.add_argument(
        "--api",
        choices=["unsplash", "pexels", "placeholder", "auto"],
        default="auto",
        help="Image API to use",
    )
    ap.add_argument("--force", action="store_true", help="Re-download existing images")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
