#!/usr/bin/env python3

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def load_jsonl_by_id(path: str) -> dict[str, dict]:
    rows = {}
    if not path:
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id") or row.get("id")
            if not sample_id:
                raise ValueError(f"Image-search row missing sample_id/id: {row}")
            rows[str(sample_id)] = row
    return rows


def as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_rel_path(path: str) -> str:
    return str(path).replace("\\", "/")


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def convert_sample(raw: dict, sample_id: str, image_rel: str, image_search: dict | None) -> dict:
    question = raw.get("query") or raw.get("question") or ""
    ground_truth = raw.get("ground_truth")
    if ground_truth is None:
        ground_truth = (raw.get("reward_model") or {}).get("ground_truth", [])
    ground_truths = as_list(ground_truth)

    out = {
        "id": sample_id,
        "prompt": [
            {
                "role": "user",
                "content": f"<image>\n{question}",
            }
        ],
        "image": [image_rel],
        "reward_model": {
            "ground_truth": ground_truths,
        },
    }

    # Preserve fields that are useful for later slicing and bad-case analysis.
    for key in ["sample_id", "query", "query_image", "difficulty", "category"]:
        if key in raw:
            out[key] = raw[key]

    # Carry through existing or externally generated reverse-image-search metadata.
    merged = dict(raw)
    if image_search:
        merged.update(image_search)

    for key in [
        "image_search_title_list",
        "image_search_thumbnail_list",
        "image_search_summary",
    ]:
        if key in merged and merged[key]:
            out[key] = merged[key]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw HR-MMSearch JSONL to SenseNova-MARS eval format."
    )
    parser.add_argument("--input-jsonl", required=True, help="Raw HR-MMSearch data.jsonl")
    parser.add_argument("--input-root", default="", help="Root for raw relative image paths")
    parser.add_argument("--output-root", required=True, help="Output directory, e.g. data/eval/hr_mmsearch")
    parser.add_argument(
        "--image-search-jsonl",
        default="",
        help="Optional sidecar JSONL keyed by sample_id with image_search_title_list/image_search_thumbnail_list",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy query images into output-root/images. If omitted, only JSON paths are rewritten.",
    )
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    input_root = Path(args.input_root) if args.input_root else input_jsonl.parent
    output_root = Path(args.output_root)
    output_images = output_root / "images"
    output_jsonl = output_root / "data.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)

    image_search_rows = load_jsonl_by_id(args.image_search_jsonl)

    count = 0
    with open(input_jsonl, encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            raw = json.loads(line)
            sample_id = str(raw.get("sample_id") or raw.get("id") or f"sample_{idx:04d}")

            image_value = raw.get("query_image")
            if image_value is None:
                image_list = as_list(raw.get("image"))
                image_value = image_list[0] if image_list else ""
            if not image_value:
                raise ValueError(f"{sample_id}: missing query_image/image")

            image_rel_in = normalize_rel_path(str(image_value))
            src_image = Path(image_rel_in)
            if not src_image.is_absolute():
                src_image = input_root / src_image

            image_rel_out = f"images/{sample_id}{src_image.suffix or '.png'}"
            if args.copy_images:
                copy_image(src_image, output_root / image_rel_out)

            row = convert_sample(
                raw=raw,
                sample_id=sample_id,
                image_rel=image_rel_out,
                image_search=image_search_rows.get(sample_id),
            )
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} samples to {output_jsonl}")
    if args.copy_images:
        print(f"Copied images to {output_images}")
    if args.image_search_jsonl:
        print(f"Merged image-search metadata for {len(image_search_rows)} samples")


if __name__ == "__main__":
    main()
