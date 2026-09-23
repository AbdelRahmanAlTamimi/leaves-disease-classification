"""
Build a sorted label map and a stratified train/val/test split
for the PlantVillage dataset (color folder).

Outputs:
  - label_map.json      : sorted {class_name: class_index}
  - split_index.json    : list of records with image_id, path, class_name,
                           class_index, split
  - Prints a per-class count table for train/val/test as a sanity check.
"""

import os
import json
import random
from collections import defaultdict
from pathlib import Path
from leaves_disease_classification.config import cfg

from sklearn.model_selection import train_test_split

# ---- Config ----
DATA_DIR = cfg.DATA_DIR  
OUTPUT_DIR = "../../../data/processed"            
SEED = cfg.SEED
TRAIN_RATIO = cfg.TRAIN_RATIO
VAL_RATIO = cfg.VAL_RATIO
TEST_RATIO = cfg.TEST_RATIO

random.seed(SEED)


def build_label_map(data_dir: Path) -> dict:
    """Sorted class name -> class index. Never trust directory listing order
    directly without sorting -- os.listdir() order is filesystem-dependent."""
    class_names = sorted(
        name for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    )
    return {name: idx for idx, name in enumerate(class_names)}


def collect_images(data_dir: Path, label_map: dict) -> list:
    """Walk the dataset and build one record per image."""
    records = []
    for class_name, class_index in label_map.items():
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            image_id = os.path.splitext(fname)[0]
            records.append({
                "image_id": image_id,
                "path": os.path.join(class_dir, fname),
                "class_name": class_name,
                "class_index": class_index,
            })
    return records


def stratified_split(records: list) -> list:
    """Split records into train/val/test, stratified by class_index,
    with a fixed seed so the split is reproducible."""
    labels = [r["class_index"] for r in records]

    train_records, temp_records, train_labels, temp_labels = train_test_split(
        records, labels,
        train_size=TRAIN_RATIO,
        stratify=labels,
        random_state=SEED,
    )

    # split temp into val/test (proportion within the remaining 30%)
    val_share = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_records, test_records = train_test_split(
        temp_records,
        train_size=val_share,
        stratify=temp_labels,
        random_state=SEED,
    )

    for r in train_records:
        r["split"] = "train"
    for r in val_records:
        r["split"] = "val"
    for r in test_records:
        r["split"] = "test"

    return train_records + val_records + test_records


def verify_split(records: list, label_map: dict) -> None:
    """Print per-class counts across splits, and raise if any class is
    missing from any split -- a silent gap here becomes a wrong prediction
    three steps later."""
    counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for r in records:
        counts[r["class_name"]][r["split"]] += 1

    print(f"{'class_name':45s} {'train':>6s} {'val':>6s} {'test':>6s}")
    missing = []
    for class_name in label_map:
        c = counts[class_name]
        print(f"{class_name:45s} {c['train']:6d} {c['val']:6d} {c['test']:6d}")
        if c["train"] == 0 or c["val"] == 0 or c["test"] == 0:
            missing.append(class_name)

    if missing:
        raise ValueError(f"Classes missing from at least one split: {missing}")
    print("\nAll classes present in train, val, and test splits.")


def main():
    label_map = build_label_map(DATA_DIR)
    assert len(label_map) == 38, f"Expected 38 classes, got {len(label_map)}"

    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    records = collect_images(DATA_DIR, label_map)
    print(f"Collected {len(records)} images across {len(label_map)} classes.")

    records = stratified_split(records)
    verify_split(records, label_map)

    with open(os.path.join(OUTPUT_DIR, "split_index.json"), "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nWrote label_map.json and split_index.json to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()