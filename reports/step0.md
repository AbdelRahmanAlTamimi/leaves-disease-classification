# Step 0 — Dataset and corruption suite

What was built, what broke, and what is still open. Numbers here come from
actual runs (`dvc repro`, `uv run pytest`) on 2026-09-23, 12-core machine,
10 worker processes.

## What exists

| Artifact | Count | Where |
|---|---|---|
| Classes | 38 | `data/label_map.json` (git-tracked, sorted) |
| Clean images | 54,305 | `data/raw/plantvillage/color/` (DVC) |
| Split index | 38,013 train / 8,146 val / 8,146 test | `data/processed/split_index.json` |
| Corrupted images | 146,628 (8,146 × 6 × 3) | `data/corrupted/` |
| Unified manifest | 200,933 records | `data/processed/manifest.json` |

Split is stratified, 70/15/15, seed 42. Every class appears in all three
splits; the smallest is `Potato___healthy` at 106 train images, comfortably
above the 50-image floor. Corruption generation: 43.3s, 3,383.6 images/s.

**Test set has not been looked at.** Corruptions are derived from the test
split by construction, but no metric has been computed on it.

## What actually broke

### 1. Directory-listing order made the split non-reproducible

`collect_images()` walked the dataset with a bare `os.listdir()`. Class
*names* were sorted, so the label map was stable — but the per-class file
listing was not, and `train_test_split` shuffles a list whose order came
straight off the filesystem. Same seed, same ratios, different split on a
different machine.

This is spec Step 0 item 4, bullet 2, and it passes every test you would
think to write: the label map is correct, the ratios are correct, every class
is present. Only a second machine exposes it.

Fix: `sorted(os.listdir(class_dir))`. The entire corrupted set was
regenerated afterwards, because the test split membership changed.

### 2. Paths were absolute

`split_index.json` stored `/home/<user>/workspace/...`, which breaks on any
other checkout and violates the project's own POSIX-relative rule. The
corruption manifest was already relative, so the two files disagreed.

Fix: `relative_to(cfg.PROJECT_ROOT).as_posix()` at write time.

### 3. One image is a PNG

54,304 JPEG + 1 PNG:

```
data/raw/plantvillage/color/Pepper,_bell___healthy/
  42f083e2-272d-4f83-ad9a-573ee90e50ec___Screen Shot 2015-05-06 at 4.01.13 PM.png
```

Mode `RGB`, bands `('R','G','B')` — no alpha channel. It landed in the
**train** split, so it never reached the corruption generator.

A full scan confirms all 54,305 clean images are mode `RGB`. The spec warns
about CMYK, greyscale and alpha; this dataset has none of them. That is a
property of the data, not of the code — see the next item.

### 4. The corruption generator would crash, not degrade, on non-RGB input

Two paths in `corruptions.py` silently assume 3-channel RGB:

- `apply_motion_blur` pads with `((0,0),(pad,pad),(0,0))` — a greyscale image
  is a 2-D array and raises on the pad.
- `apply_jpeg_compression` saves as JPEG — an RGBA image raises
  `OSError: cannot write mode RGBA as JPEG`.

It ran clean over all 8,146 test images, which is evidence the test split is
uniformly RGB, not a guarantee for future data. There is no
`.convert("RGB")` call anywhere in `src/` yet. It belongs in both the
training and serving paths when those are built — user uploads are the real
threat, not this dataset.

### 5. Schema drift against the spec

Three deviations caught on review, all in the manifest:

- Clean records used `"corruption": "none"`; the spec schema uses `null`.
- The `species` field (→ `plant` here) was missing entirely.
- Field order did not match the spec.

None of these would have failed a test that was written against the
implementation instead of against the spec. The tests now assert
`corruption is None`, and the label-map test pins all 38 names to indices
0–37 against a hardcoded reference rather than just checking sortedness.

## Validation

`uv run pytest` — 7 tests, 30.04s. All five assertions required by spec
Step 0 item 5 are covered:

| Assertion | Test |
|---|---|
| Every path exists and opens | `test_all_manifest_paths_exist_and_open` (PIL `verify()`, no pixel decode) |
| No `image_id` in two splits | `test_no_image_id_in_more_than_one_split` |
| Label map = 38 entries, matches reference | `test_label_map_matches_the_committed_reference` |
| ≥ 50 training images per class | `test_every_class_has_enough_training_images` |
| No image under 32×32 | `test_no_image_is_smaller_than_32x32` |

Plus `test_every_class_appears_in_every_split` and
`test_every_test_image_has_a_full_corruption_set` (18 records per test image).

The path/open check covers all 200,933 records and accounts for ~20s of the
30s runtime. If that becomes a problem in CI, the lever is verifying clean
images only.

## Open items

- **No DVC remote.** `dvc status` is clean and the pipeline reproduces
  locally, but rubric row 05 passes on `git checkout` + `dvc pull` +
  `dvc repro`, and a reviewer cannot pull today. Only the 918 MB raw dataset
  needs pushing — `data/corrupted` (3.1 GB) regenerates in 43s and the
  manifests in seconds. Deferred to Session 2.
- **`manifest.json` is DVC-tracked, not git-committed.** It is 85 MB. Spec
  item 3 says "committed as JSON and DVC-tracked"; checklist line 120 says
  tracked with DVC. `label_map.json` (1.3 KB) *is* git-committed, which is
  what checklist line 106 requires.
- **`corruptions.py` lives in `src/leaves_disease_classification/data/`**,
  not `data/` as spec item 2 states. CLAUDE.md's src-layout rule takes
  precedence, noted here because a reviewer may grep for the literal path.

## Deferred to later sessions

Spec item 4 lists six hazards. Two were hit and are written up above
(listing order, image modes). The remaining four need artifacts that do not
exist yet and are carried forward:

- Train/serve preprocessing skew — export the eval transform with the
  weights, load from one place, assert logits agree to 1e-4 (Session 1).
- Confusable classes — PlantVillage's own confusions are the Tomato group
  and Early vs Late blight across Potato and Tomato, not the spec's
  spaniels. Confusion matrix justifies the abstention threshold (Session 4).
- Softmax is not calibrated — temperature scaling on the validation fold,
  reliability diagram before and after (Session 4).
- Augmentation in the training path only (Session 1).

Spec item 7 (`/predict` abstention, top-3, coverage and selective accuracy)
is API work and is not startable in Step 0.
