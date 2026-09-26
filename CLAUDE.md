# CLAUDE.md

## Project

MLOps course final project, Track C (Computer Vision): plant disease classification.
The full requirements are in `docs/project_spec.md` — read it before starting any task.

The spec is written for Oxford-IIIT Pet. This project uses **PlantVillage** instead
(an approved alternative). Every rule in the spec applies unchanged, with these substitutions:

| Spec (Oxford-IIIT Pet) | This project (PlantVillage) |
|---|---|
| 37 breeds | 38 classes (`plant___condition`, e.g. `Tomato___Late_blight`) |
| `breed` | `class_name` in data files; `disease` in the API (e.g. `Late_blight`, `healthy`) |
| `species` | `plant` (the part of `class_name` before `___`, e.g. `Tomato`) |
| trainval/test shipped split | our own stratified 70/15/15 split, seed 42 |
| `PetBreedClassifier` | `PlantDiseaseClassifier` |
| `docker build -t pet-breed .` | `docker build -t plant-disease .` |

Dataset: Kaggle PlantVillage, `color` folder only — 38 classes, 54,305 images.
Some plants have only a healthy class (Blueberry, Raspberry, Soybean) and Squash has
only a disease class, so the label space is not symmetric per plant.

## Rules

- KISS and YAGNI: simplest working solution. No abstractions, options or files that the
  current task doesn't need.
- All code, comments and names in English.
- Package manager is `uv` only — never pip. `uv add` / `uv add --dev`.
- Package lives in `src/leaves_disease_classification/`. Run modules with
  `uv run python -m leaves_disease_classification.<module>`.
- Shared paths and params come from `cfg` in `config.py`.
- Paths stored in JSON files are relative to the project root, POSIX form.
- Never look at test-set metrics before Session 4 work (see spec, Step 0.1).
- Report real numbers only (see the honesty clause in the spec).
- Don't create git commits; the user reviews and commits.
