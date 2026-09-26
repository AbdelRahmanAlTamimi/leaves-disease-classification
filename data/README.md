# Data

## Source

[PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
(Kaggle), `color` folder only. The `grayscale` and `segmented` folders are not used.

- **38 classes** (crop + disease, e.g. `Tomato___Late_blight`, `Apple___healthy`)
- **54,305 images**, 256x256 JPEG
- Location: `data/raw/plantvillage/color/<class_name>/<image_id>.JPG`

## Split

Stratified by class, **70% train / 15% val / 15% test**, `seed = 42`
(`cfg.SEED`). Every class is present in all three splits; the stage fails
if one is not. Test split: **8,146 images**.

The split index lives in `data/processed/split_index.json`, the class
name -> index mapping in `data/label_map.json` (sorted class names,
indices 0..37).

## Corruption suite

Applied to the **test split only**, every corruption at 3 severities:
8,146 x 6 x 3 = **146,628 images**. This is the labeled drift ground truth
used later to score drift detectors against a known answer.

| Corruption | What it does | Severity 1 | Severity 2 | Severity 3 |
|---|---|---|---|---|
| `gaussian_blur` | Gaussian blur, out-of-focus camera | radius 1 | radius 2 | radius 4 |
| `brightness_up` | Overexposure | factor 1.3 | factor 1.6 | factor 2.0 |
| `brightness_down` | Underexposure | factor 0.7 | factor 0.5 | factor 0.3 |
| `jpeg_compression` | Re-encode with heavy compression artifacts | quality 50 | quality 30 | quality 10 |
| `downscale_upscale` | Downscale then upscale back to 256x256, losing detail | to 128px | to 96px | to 64px |
| `motion_blur` | Horizontal moving average, camera shake | kernel 5 | kernel 9 | kernel 15 |

Corrupted images are written to
`data/corrupted/<corruption>/severity_<n>/<class_name>/<image_id>.jpg`.

## Output files

| File | Contents |
|---|---|
| `data/label_map.json` | `{class_name: class_index}`, sorted, 38 entries. Git-tracked. |
| `data/processed/split_index.json` | One record per clean image: `image_id, path, class_name, class_index, split`. |
| `data/processed/corruption_manifest.json` | One record per corrupted image, with `corruption`, `severity`, `width`, `height`. |
| `data/processed/manifest.json` | Clean + corrupted merged, **200,933 records**, fields: `image_id, path, class_name, plant, class_index, split, corruption, severity, width, height`. Clean images have `corruption = null` and `severity = 0`. |

All paths in these files are relative to the project root, in POSIX form.
`plant` is the part of `class_name` before `___`, kept raw (e.g. `Corn_(maize)`).

```json
{
  "image_id": "0a1b2c3d-...___RS_Late.B 7111",
  "path": "data/raw/plantvillage/color/Tomato___Late_blight/0a1b2c3d-...___RS_Late.B 7111.JPG",
  "class_name": "Tomato___Late_blight",
  "plant": "Tomato",
  "class_index": 30,
  "split": "test",
  "corruption": null,
  "severity": 0,
  "width": 256,
  "height": 256
}
```

## Rebuild

```bash
dvc repro     # split -> corruptions -> manifest
uv run pytest # pre-training checks on the generated files
```

The raw dataset is DVC-tracked (`data/raw/plantvillage/color.dvc`); it is not
in git. With no DVC remote configured, restore it by downloading the dataset
from Kaggle into `data/raw/plantvillage/color/`.
