"""
Pre-training data checks. These run against the generated files, not against
the code that produced them: if the pipeline is stale, half-finished, or was
run against the wrong dataset, these tests fail before any training starts.
"""

import json
from collections import Counter, defaultdict

import pytest
from PIL import Image

from leaves_disease_classification.config import cfg

MIN_TRAIN_IMAGES_PER_CLASS = 50
MIN_IMAGE_SIZE = 32
CORRUPTED_RECORDS_PER_TEST_IMAGE = 18  # 6 corruptions x 3 severities

# The committed reference label space. Class indices must equal the position
# in this list -- a reordered label map is the bug that makes the API answer
# confidently and wrongly for every image.
EXPECTED_CLASS_NAMES = (
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(cfg.MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def label_map():
    return json.loads(cfg.LABEL_MAP_PATH.read_text())


def test_all_manifest_paths_exist_and_open(manifest):
    missing = []
    unreadable = []
    for r in manifest:
        path = cfg.PROJECT_ROOT / r["path"]
        if not path.exists():
            missing.append(r["path"])
            continue
        try:
            # verify() checks the file structure without decoding pixels
            with Image.open(path) as img:
                img.verify()
        except Exception as exc:
            unreadable.append(f"{r['path']}: {exc}")

    assert not missing, f"{len(missing)} manifest paths do not exist, e.g. {missing[:5]}"
    assert not unreadable, f"{len(unreadable)} manifest images do not open, e.g. {unreadable[:5]}"


def test_no_image_is_smaller_than_32x32(manifest):
    too_small = [
        (r["path"], r["width"], r["height"]) for r in manifest
        if r["width"] < MIN_IMAGE_SIZE or r["height"] < MIN_IMAGE_SIZE
    ]
    assert not too_small, (
        f"{len(too_small)} images are smaller than "
        f"{MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}, e.g. {too_small[:5]}"
    )


def test_no_image_id_in_more_than_one_split(manifest):
    splits_by_id = defaultdict(set)
    for r in manifest:
        if r["corruption"] is None:
            splits_by_id[r["image_id"]].add(r["split"])

    leaked = {i: s for i, s in splits_by_id.items() if len(s) > 1}
    assert not leaked, f"{len(leaked)} image_ids appear in several splits, e.g. {list(leaked)[:5]}"


def test_label_map_matches_the_committed_reference(label_map):
    assert len(label_map) == len(EXPECTED_CLASS_NAMES)
    expected = {name: index for index, name in enumerate(EXPECTED_CLASS_NAMES)}
    assert label_map == expected


def test_every_class_appears_in_every_split(manifest, label_map):
    classes_by_split = defaultdict(set)
    for r in manifest:
        if r["corruption"] is None:
            classes_by_split[r["split"]].add(r["class_name"])

    for split in ("train", "val", "test"):
        missing = set(label_map) - classes_by_split[split]
        assert not missing, f"classes missing from {split}: {sorted(missing)}"


def test_every_class_has_enough_training_images(manifest):
    counts = Counter(
        r["class_name"] for r in manifest
        if r["corruption"] is None and r["split"] == "train"
    )
    too_small = {c: n for c, n in counts.items() if n < MIN_TRAIN_IMAGES_PER_CLASS}
    assert not too_small, f"classes with fewer than {MIN_TRAIN_IMAGES_PER_CLASS} train images: {too_small}"


def test_every_test_image_has_a_full_corruption_set(manifest):
    clean_test_ids = {
        r["image_id"] for r in manifest
        if r["corruption"] is None and r["split"] == "test"
    }
    corrupted_counts = Counter(
        r["image_id"] for r in manifest if r["corruption"] is not None
    )

    wrong = {
        image_id: corrupted_counts[image_id]
        for image_id in clean_test_ids
        if corrupted_counts[image_id] != CORRUPTED_RECORDS_PER_TEST_IMAGE
    }
    assert not wrong, (
        f"{len(wrong)} test images do not have exactly "
        f"{CORRUPTED_RECORDS_PER_TEST_IMAGE} corrupted records, e.g. {list(wrong.items())[:5]}"
    )
    assert set(corrupted_counts) == clean_test_ids, "corrupted records reference non-test images"
