#!/usr/bin/env python3
"""
Automated diverse image picker from benchmark-source-images S3 bucket.

1. Scans all adobe scene categories in S3
2. Maps them to our benchmark groups (person / object / scene / style)
3. Picks a diverse set matching target distribution
4. Downloads, resizes, and saves a pre-filter manifest for review

Usage:
    python scripts/pick_diverse_images.py                # preview only (dry run)
    python scripts/pick_diverse_images.py --download      # download selected images
    python scripts/pick_diverse_images.py --download --total 165  # custom total
"""

import argparse
import boto3
import json
import os
import random
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

ENDPOINT = "https://s3.us-east-005.backblazeb2.com"
BUCKET = "benchmark-source-images"
ACCESS_KEY = "005bd09615863760000000002"
SECRET_KEY = "K005M/xHme7keBbhzSVl2SqM2LmUHwc"
REGION = "us-east-005"

SCENE_PREFIX = "images/adobe/"
PERSON_PREFIX = "adobe-human-image-editing/accepted/"

OUT_DIR = Path(__file__).resolve().parent.parent / "prompts" / "source_images"

# Map S3 categories → our benchmark groups
CATEGORY_MAP = {
    # OBJECT categories (~30%)
    "object": [
        "vehicles_cars",
        "vehicles_motorcycles",
        "vehicles_bicycles",
        "vehicles_boats_ships",
        "vehicles_aircraft",
        "vehicles_trains",
        "vehicles_trucks_heavy",
        "food_desserts_sweets",
        "food_prepared_meals",
        "food_street_food",
        "food_ingredients",
        "beverages_hot",
        "beverages_cold",
        "beverages_alcoholic",
        "fruits_vegetables",
        "cosmetics_beauty",
        "fashion_clothing",
        "shoes_footwear",
        "hats_headwear",
        "kitchen_cookware",
        "tools_hardware",
        "gardening_tools",
        "toys_games",
        "music_instruments",
        "technology_circuits_hardware",
        "technology_wearables",
        "technology_robotics",
        "flowers_roses",
        "flowers_tropical",
        "flowers_wildflowers",
        "flowers_arrangements",
        "plants_succulents_cacti",
        "art_sculpture",
        "art_photography_equipment",
    ],
    # SCENE categories (~20%)
    "scene": [
        "architecture_modern",
        "architecture_classical",
        "architecture_bridges",
        "architecture_skyscrapers",
        "architecture_ruins_ancient",
        "architecture_religious",
        "urban_street_scenes",
        "urban_night_city",
        "urban_markets_bazaars",
        "landscape_mountains",
        "landscape_forest",
        "landscape_tropical",
        "landscape_arctic_tundra",
        "landscape_meadow_plains",
        "waterscape_lakes",
        "waterscape_rivers_streams",
        "marine_nautical",
        "workspace_desk",
        "restaurant_dining",
        "caf_coffee_shop",
        "hotel_hospitality",
        "retail_shopping",
        "fitness_gym",
        "cinema_theater",
        "construction_building",
        "industrial_manufacturing",
        "medical_healthcare",
        "science_laboratory",
        "education_classroom",
        "camping_outdoors",
        "playground_recreation",
    ],
    # STYLE categories (~10%) — scenes good for style/global edits
    "style": [
        "art_painting",
        "holiday_christmas",
        "holiday_halloween",
        "holiday_general_celebrations",
        "dance_performance",
        "music_performance",
        "gardens_landscaping",
        "trees_foliage",
        "rural_countryside",
        "energy_renewable",
    ],
}

# Person images come from the existing adobe-human-image-editing prefix
PERSON_SUBCATS = ["portrait", "fullbody", "group"]

TARGET_DIST = {
    "person": 0.40,
    "object": 0.30,
    "scene": 0.20,
    "style": 0.10,
}


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
    )


def load_pics_json(s3, category: str) -> list[dict]:
    key = f"{SCENE_PREFIX}{category}/pics.json"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        data = json.loads(obj["Body"].read())
        return data.get("images", [])
    except Exception:
        return []


def get_full_scenes(images: list[dict]) -> list[dict]:
    """Return only full_scene shots (shot01), not isolation shots."""
    return [img for img in images if "full_scene" in img.get("filename", "")]


def list_person_images(s3, subcat: str, limit: int = 200) -> list[str]:
    prefix = f"{PERSON_PREFIX}{subcat}/"
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=BUCKET, Prefix=prefix, PaginationConfig={"MaxItems": limit}
    ):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if any(k.lower().endswith(e) for e in [".jpg", ".jpeg", ".png", ".webp"]):
                keys.append(k)
    return keys


def pick_diverse_set(s3, total: int = 165) -> dict:
    """Pick a diverse image set matching target distribution."""
    selection = {"person": [], "object": [], "scene": [], "style": []}

    counts = {group: max(1, int(total * pct)) for group, pct in TARGET_DIST.items()}
    # Adjust rounding
    diff = total - sum(counts.values())
    if diff > 0:
        counts["person"] += diff
    elif diff < 0:
        counts["person"] = max(1, counts["person"] + diff)

    print(f"Target: {counts}")
    print()

    # --- Person images ---
    print("Scanning person images...")
    person_pool = []
    for subcat in PERSON_SUBCATS:
        keys = list_person_images(s3, subcat, limit=500)
        for k in keys:
            person_pool.append(
                {
                    "s3_key": k,
                    "group": "person",
                    "category": f"person_{subcat}",
                    "description": f"{subcat} portrait from hub production data",
                    "filename": k.split("/")[-1],
                }
            )
    random.shuffle(person_pool)
    selection["person"] = person_pool[: counts["person"]]
    print(f"  Found {len(person_pool)} person images, picked {len(selection['person'])}")

    # --- Object / Scene / Style images ---
    for group in ["object", "scene", "style"]:
        print(f"\nScanning {group} categories...")
        categories = CATEGORY_MAP[group]
        random.shuffle(categories)
        per_cat = max(1, counts[group] // len(categories)) + 1
        pool = []

        for cat in categories:
            imgs = load_pics_json(s3, cat)
            scenes = get_full_scenes(imgs)
            if not scenes:
                continue
            random.shuffle(scenes)
            for img in scenes[:per_cat]:
                pool.append(
                    {
                        "s3_key": f"{SCENE_PREFIX}{cat}/{img['filename']}",
                        "group": group,
                        "category": cat,
                        "description": img.get("description", cat),
                        "filename": img["filename"],
                    }
                )

        random.shuffle(pool)
        # Deduplicate by category — pick at most 2 per category for diversity
        seen_cats = defaultdict(int)
        diverse_pool = []
        for item in pool:
            if seen_cats[item["category"]] < 2:
                diverse_pool.append(item)
                seen_cats[item["category"]] += 1

        selection[group] = diverse_pool[: counts[group]]
        print(
            f"  Scanned {len(categories)} categories, {len(pool)} full scenes, picked {len(selection[group])}"
        )

    return selection


def download_and_resize(s3, s3_key: str, dest_path: Path, max_dim: int = 1024):
    """Download from S3, resize to max_dim, save as JPEG."""
    obj = s3.get_object(Bucket=BUCKET, Key=s3_key)
    raw = obj["Body"].read()

    if Image is None:
        dest_path.write_bytes(raw)
        return

    img = Image.open(BytesIO(raw))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img.save(dest_path, "JPEG", quality=90)


def main():
    parser = argparse.ArgumentParser(description="Pick diverse benchmark images from hub S3")
    parser.add_argument("--total", type=int, default=165, help="Total images to pick")
    parser.add_argument(
        "--download", action="store_true", help="Actually download (otherwise dry run)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    s3 = get_s3_client()

    print(f"=== Diverse Image Picker (total={args.total}, seed={args.seed}) ===\n")
    selection = pick_diverse_set(s3, total=args.total)

    # Print summary
    print("\n" + "=" * 70)
    print("SELECTION SUMMARY")
    print("=" * 70)
    all_items = []
    for group in ["person", "object", "scene", "style"]:
        items = selection[group]
        all_items.extend(items)
        cats = set(i["category"] for i in items)
        print(f"\n{group.upper()}: {len(items)} images from {len(cats)} categories")
        for cat in sorted(cats):
            cat_items = [i for i in items if i["category"] == cat]
            print(f"  {cat}: {len(cat_items)}")
            for it in cat_items[:1]:
                print(f"    → {it['description'][:80]}")

    print(f"\nTOTAL: {len(all_items)} images")

    # Save manifest
    manifest_path = OUT_DIR / "diversity_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "total": len(all_items),
        "distribution": {g: len(selection[g]) for g in selection},
        "images": all_items,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")

    if not args.download:
        print("\n⚠ DRY RUN — pass --download to actually fetch images")
        return

    # Download
    print(f"\nDownloading {len(all_items)} images...")
    for group in ["person", "object", "scene", "style"]:
        group_dir = OUT_DIR / group
        group_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(all_items):
        group_dir = OUT_DIR / item["group"]
        ext = "jpg"
        dest = group_dir / f"{item['group']}_{item['category']}_{i:04d}.{ext}"
        item["local_path"] = str(dest.relative_to(OUT_DIR.parent.parent))
        try:
            download_and_resize(s3, item["s3_key"], dest)
            status = "✓"
        except Exception as e:
            status = f"✗ {e}"
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i + 1}/{len(all_items)}] {status} {dest.name}")

    # Update manifest with local paths
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone! Images saved to {OUT_DIR}/[person|object|scene|style]/")
    print(f"Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
