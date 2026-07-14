#!/usr/bin/env python3
"""
Scan benchmark-source-images S3 bucket, discover all top-level prefixes,
and report what image categories exist beyond adobe-human-image-editing.
"""
import boto3
from collections import Counter

ENDPOINT = "https://s3.us-east-005.backblazeb2.com"
BUCKET = "benchmark-source-images"
ACCESS_KEY = "005bd09615863760000000002"
SECRET_KEY = "K005M/xHme7keBbhzSVl2SqM2LmUHwc"
REGION = "us-east-005"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

def main():
    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
    )

    # List top-level "directories" (common prefixes with delimiter)
    print("=== Top-level prefixes in benchmark-source-images ===\n")
    resp = s3.list_objects_v2(Bucket=BUCKET, Delimiter="/", MaxKeys=1000)
    prefixes = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
    for p in sorted(prefixes):
        print(f"  {p}")

    print(f"\nTotal top-level prefixes: {len(prefixes)}")

    # For each prefix, drill one level deeper and count images
    print("\n=== Image counts per prefix (sampling first 500 keys each) ===\n")
    results = []
    for prefix in sorted(prefixes):
        paginator = s3.get_paginator("list_objects_v2")
        count = 0
        sub_prefixes = set()
        sample_keys = []
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, MaxKeys=500):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                ext = "." + key.rsplit(".", 1)[-1].lower() if "." in key else ""
                if ext in IMAGE_EXTS:
                    count += 1
                    if len(sample_keys) < 3:
                        sample_keys.append(key)
                # Track sub-prefixes (second level)
                remainder = key[len(prefix):]
                if "/" in remainder:
                    sub_prefixes.add(remainder.split("/")[0])
            break  # only first page (500 keys) for speed

        if count > 0 or sub_prefixes:
            results.append((prefix, count, sub_prefixes, sample_keys))
            status = f"{count} images" if count > 0 else "no images"
            subs = ", ".join(sorted(sub_prefixes)[:10]) if sub_prefixes else "flat"
            print(f"  {prefix:<50} {status:<15} sub: [{subs}]")
            for sk in sample_keys:
                print(f"    sample: {sk}")

    # Also check for accepted/rejected structure
    print("\n=== Checking known prefixes with accepted/ subdirs ===\n")
    known = [p for p in prefixes if any(kw in p.lower() for kw in
             ["adobe", "edit", "image", "photo", "product", "object", "scene"])]

    if not known:
        print("  No obvious image-editing prefixes found. Listing all with images:")
        known = [r[0] for r in results if r[1] > 0]

    for prefix in known:
        # Check for accepted/ subdir
        resp2 = s3.list_objects_v2(
            Bucket=BUCKET, Prefix=f"{prefix}accepted/", MaxKeys=10
        )
        n = resp2.get("KeyCount", 0)
        print(f"  {prefix}accepted/ → {n} objects (sampled)")

        resp3 = s3.list_objects_v2(
            Bucket=BUCKET, Prefix=f"{prefix}rejected/", MaxKeys=10
        )
        n2 = resp3.get("KeyCount", 0)
        print(f"  {prefix}rejected/ → {n2} objects (sampled)")

if __name__ == "__main__":
    main()
