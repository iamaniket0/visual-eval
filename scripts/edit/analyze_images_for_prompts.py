"""Build image analysis metadata from manifest descriptions + category info.

For object/scene/style images: uses the rich S3 descriptions from pics.json.
For person images: generates structured metadata from category (portrait/fullbody/group).

This produces image_analysis.json consumed by the prompt generator.

Usage:
    python scripts/analyze_images_for_prompts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = ROOT / "prompts" / "source_images" / "diversity_manifest.json"
OUTPUT_PATH = ROOT / "prompts" / "source_images" / "image_analysis.json"

PERSON_TEMPLATES = {
    "person_portrait": {
        "scene_type": "studio/outdoor",
        "primary_subject": "person portrait (head and shoulders)",
        "has_people": True,
        "editable_regions": [
            "hair color/style",
            "eye color",
            "skin tone",
            "facial expression",
            "clothing/top",
            "accessories (glasses, jewelry, hat)",
            "background behind subject",
            "lighting/mood",
        ],
        "typical_objects": ["clothing", "accessories", "background elements"],
    },
    "person_fullbody": {
        "scene_type": "outdoor/indoor",
        "primary_subject": "person full body shot",
        "has_people": True,
        "editable_regions": [
            "hair color/style",
            "top/shirt",
            "bottom/pants/skirt",
            "shoes/footwear",
            "accessories (bag, watch, hat)",
            "pose/stance",
            "background/environment",
            "ground surface",
            "lighting/shadows",
        ],
        "typical_objects": [
            "clothing items",
            "footwear",
            "bag/accessories",
            "ground",
            "background structures",
        ],
    },
    "person_group": {
        "scene_type": "outdoor/indoor",
        "primary_subject": "group of people",
        "has_people": True,
        "editable_regions": [
            "specific person's clothing",
            "add/remove a person",
            "background environment",
            "group arrangement",
            "lighting/time of day",
            "foreground objects",
            "individual accessories",
            "ground surface",
        ],
        "typical_objects": ["multiple people", "clothing items", "background structures", "ground"],
    },
}


def parse_description(desc: str) -> dict:
    """Extract structured info from S3 scene descriptions."""
    objects = []
    materials = []

    parts = desc.split(" with ")
    scene_part = parts[0].strip()

    if len(parts) > 1:
        obj_parts = " with ".join(parts[1:])
        for item in obj_parts.split(", "):
            item = item.strip()
            if item:
                objects.append(item)

    has_people = any(
        w in desc.lower()
        for w in [
            "person",
            "people",
            "man",
            "woman",
            "child",
            "worker",
            "chef",
            "farmer",
            "tourist",
            "dancer",
            "musician",
        ]
    )

    for mat_word in [
        "wooden",
        "metal",
        "glass",
        "stone",
        "brick",
        "concrete",
        "marble",
        "steel",
        "plastic",
        "leather",
        "fabric",
        "ceramic",
        "bronze",
        "copper",
        "aluminum",
        "iron",
    ]:
        if mat_word in desc.lower():
            materials.append(mat_word)

    return {
        "scene_type": "outdoor"
        if any(
            w in desc.lower()
            for w in [
                "outdoor",
                "street",
                "garden",
                "farm",
                "beach",
                "mountain",
                "lake",
                "river",
                "forest",
                "park",
                "road",
                "highway",
            ]
        )
        else "indoor",
        "primary_subject": scene_part[:100],
        "has_people": has_people,
        "objects": objects[:15],
        "materials": materials,
        "description_full": desc,
    }


def build_analysis(img: dict) -> dict:
    group = img["group"]
    category = img["category"]
    desc = img.get("description", "")

    if group == "person":
        template = PERSON_TEMPLATES.get(category, PERSON_TEMPLATES["person_fullbody"])
        return {
            **template,
            "group": group,
            "category": category,
            "local_path": img["local_path"],
        }
    else:
        parsed = parse_description(desc)
        editable = []
        if parsed["objects"]:
            editable.append(f"replace/modify {parsed['objects'][0]}")
        editable.append("background/environment change")
        editable.append("lighting/time of day change")
        editable.append("color/material of main subject")
        if parsed["has_people"]:
            editable.append("person's clothing/appearance")
        editable.append("add new object to scene")
        editable.append("style transfer (artistic/era)")
        if len(parsed["objects"]) > 2:
            editable.append("remove specific object")

        return {
            **parsed,
            "editable_regions": editable[:8],
            "group": group,
            "category": category,
            "local_path": img["local_path"],
        }


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    images = manifest["images"]

    results = {}
    for img in images:
        local = ROOT / img["local_path"]
        if not local.exists():
            print(f"  SKIP (missing): {img['local_path']}")
            continue
        results[img["local_path"]] = build_analysis(img)

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Done! {len(results)} images analyzed -> {OUTPUT_PATH}")

    by_group = {}
    for v in results.values():
        g = v["group"]
        by_group[g] = by_group.get(g, 0) + 1
    for g, c in sorted(by_group.items()):
        print(f"  {g}: {c}")


if __name__ == "__main__":
    main()
