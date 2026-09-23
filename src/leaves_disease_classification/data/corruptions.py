"""
Corruption suite for PlantVillage -- generates a corrupted copy of the
test split at 3 severity levels for each corruption type. This is the
project's labeled drift ground truth: we know exactly which corruption
and severity was applied to each generated image, so drift detectors
built later can be scored against a known answer instead of guessed.

Reads:  split_index.json (produced by build_dataset_split.py)
Writes: <OUTPUT_ROOT>/<corruption_type>/severity_<n>/<class_name>/<image_id>.jpg
        corruption_manifest.json -- one record per generated image, with
        the same shape as split_index.json plus corruption/severity fields.
"""

import os
import io
import json

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# ---- Config ----
SPLIT_INDEX_PATH = "split_index.json"   # EDIT: path to the file from build_dataset_split.py
OUTPUT_ROOT = "corrupted_test"          # EDIT: where corrupted images are written
MANIFEST_OUT = "corruption_manifest.json"

# Severity levels 1 (mild) -> 3 (severe) for each corruption type.
GAUSSIAN_BLUR_RADIUS = [1, 2, 4]
BRIGHTNESS_UP_FACTOR = [1.3, 1.6, 2.0]
BRIGHTNESS_DOWN_FACTOR = [0.7, 0.5, 0.3]
JPEG_QUALITY = [50, 30, 10]
DOWNSCALE_UPSCALE_SIZE = [128, 96, 64]
MOTION_BLUR_KERNEL_SIZE = [5, 9, 15]


def load_test_records(split_index_path: str) -> list:
    with open(split_index_path) as f:
        records = json.load(f)
    test_records = [r for r in records if r["split"] == "test"]
    print(f"Loaded {len(test_records)} test images to corrupt.")
    return test_records


def open_as_rgb(path: str) -> Image.Image:
    """Force RGB here specifically because this script re-encodes as JPEG
    and applies pixel-level filters that assume 3 channels -- this is a
    data-generation necessity, independent of any RGB policy in the
    serving API."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def apply_gaussian_blur(img: Image.Image, severity: int) -> Image.Image:
    radius = GAUSSIAN_BLUR_RADIUS[severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_brightness_up(img: Image.Image, severity: int) -> Image.Image:
    factor = BRIGHTNESS_UP_FACTOR[severity - 1]
    return ImageEnhance.Brightness(img).enhance(factor)


def apply_brightness_down(img: Image.Image, severity: int) -> Image.Image:
    factor = BRIGHTNESS_DOWN_FACTOR[severity - 1]
    return ImageEnhance.Brightness(img).enhance(factor)


def apply_jpeg_compression(img: Image.Image, severity: int) -> Image.Image:
    quality = JPEG_QUALITY[severity - 1]
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_downscale_upscale(img: Image.Image, severity: int) -> Image.Image:
    size = DOWNSCALE_UPSCALE_SIZE[severity - 1]
    original_size = img.size
    small = img.resize((size, size), Image.BILINEAR)
    return small.resize(original_size, Image.BILINEAR)


def apply_motion_blur(img: Image.Image, severity: int) -> Image.Image:
    kernel_size = MOTION_BLUR_KERNEL_SIZE[severity - 1]
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0
    kernel = kernel / kernel.sum()

    arr = np.array(img).astype(np.float32)
    blurred = np.zeros_like(arr)
    pad = kernel_size // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            region = padded[i:i + kernel_size, j:j + kernel_size, :]
            blurred[i, j, :] = np.tensordot(kernel, region, axes=([0, 1], [0, 1]))
    return Image.fromarray(blurred.astype(np.uint8))


CORRUPTIONS = {
    "gaussian_blur": apply_gaussian_blur,
    "brightness_up": apply_brightness_up,
    "brightness_down": apply_brightness_down,
    "jpeg_compression": apply_jpeg_compression,
    "downscale_upscale": apply_downscale_upscale,
    "motion_blur": apply_motion_blur,
}


def generate_corrupted_set(test_records: list) -> list:
    output_records = []
    total = len(test_records) * len(CORRUPTIONS) * 3
    done = 0

    for corruption_name, corruption_fn in CORRUPTIONS.items():
        for severity in (1, 2, 3):
            out_dir_base = os.path.join(
                OUTPUT_ROOT, corruption_name, f"severity_{severity}"
            )
            for record in test_records:
                img = open_as_rgb(record["path"])
                corrupted_img = corruption_fn(img, severity)

                class_dir = os.path.join(out_dir_base, record["class_name"])
                os.makedirs(class_dir, exist_ok=True)
                out_path = os.path.join(class_dir, f"{record['image_id']}.jpg")
                corrupted_img.save(out_path, format="JPEG", quality=95)

                output_records.append({
                    "image_id": record["image_id"],
                    "path": out_path,
                    "class_name": record["class_name"],
                    "class_index": record["class_index"],
                    "split": "test",
                    "corruption": corruption_name,
                    "severity": severity,
                    "width": corrupted_img.width,
                    "height": corrupted_img.height,
                })

                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{total} corrupted images generated")

    return output_records


def main():
    test_records = load_test_records(SPLIT_INDEX_PATH)
    print(
        f"Will generate {len(test_records) * len(CORRUPTIONS) * 3} corrupted "
        f"images ({len(CORRUPTIONS)} corruption types x 3 severities x "
        f"{len(test_records)} test images). This may take a while."
    )

    output_records = generate_corrupted_set(test_records)

    with open(MANIFEST_OUT, "w") as f:
        json.dump(output_records, f, indent=2)

    print(f"\nWrote {len(output_records)} corrupted-image records to {MANIFEST_OUT}")


if __name__ == "__main__":
    main()