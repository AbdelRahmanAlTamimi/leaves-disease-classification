"""
Merge the clean split index and the corruption manifest into one unified
manifest: one record per image, clean and corrupted alike.

Clean images carry corruption = null and severity = 0, so a single file
answers "what is this image, where is it, and what was done to it" for the
whole dataset.

Reads:  cfg.SPLIT_INDEX_PATH, cfg.CORRUPTION_MANIFEST_PATH
Writes: cfg.MANIFEST_PATH

Run:    uv run python -m leaves_disease_classification.data.build_manifest
"""

import json

from PIL import Image

from leaves_disease_classification.config import cfg


def plant_of(class_name: str) -> str:
    """Class names are "<plant>___<condition>"; the raw plant name is kept
    as-is, e.g. "Corn_(maize)"."""
    return class_name.split("___")[0]


def read_size(relative_path: str) -> tuple[int, int]:
    """Image dimensions from the header only -- PIL does not decode pixels
    until they are accessed."""
    with Image.open(cfg.PROJECT_ROOT / relative_path) as img:
        return img.width, img.height


def clean_records() -> list[dict]:
    records = json.loads(cfg.SPLIT_INDEX_PATH.read_text())
    out = []
    for i, r in enumerate(records, start=1):
        width, height = read_size(r["path"])
        out.append({
            "image_id": r["image_id"],
            "path": r["path"],
            "class_name": r["class_name"],
            "plant": plant_of(r["class_name"]),
            "class_index": r["class_index"],
            "split": r["split"],
            "corruption": None,
            "severity": 0,
            "width": width,
            "height": height,
        })
        if i % 10000 == 0 or i == len(records):
            print(f"  {i}/{len(records)} clean images measured")
    return out


def corrupted_records() -> list[dict]:
    records = json.loads(cfg.CORRUPTION_MANIFEST_PATH.read_text())
    return [{
        "image_id": r["image_id"],
        "path": r["path"],
        "class_name": r["class_name"],
        "plant": plant_of(r["class_name"]),
        "class_index": r["class_index"],
        "split": r["split"],
        "corruption": r["corruption"],
        "severity": r["severity"],
        "width": r["width"],
        "height": r["height"],
    } for r in records]


def main() -> None:
    clean = clean_records()
    corrupted = corrupted_records()
    manifest = clean + corrupted

    cfg.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print(f"\nClean records:     {len(clean)}")
    print(f"Corrupted records: {len(corrupted)}")
    print(f"Wrote {len(manifest)} records to {cfg.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
