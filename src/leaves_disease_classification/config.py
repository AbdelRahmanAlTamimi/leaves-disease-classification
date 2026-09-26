from dataclasses import dataclass
from pathlib import Path

# src/leaves_disease_classification/config.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Config:
    # Paths
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = DATA_ROOT / "raw" / "plantvillage" / "color"
    LABEL_MAP_PATH: Path = DATA_ROOT / "label_map.json"
    SPLIT_INDEX_PATH: Path = DATA_ROOT / "processed" / "split_index.json"
    CORRUPTED_DIR: Path = DATA_ROOT / "corrupted"
    CORRUPTION_MANIFEST_PATH: Path = DATA_ROOT / "processed" / "corruption_manifest.json"
    MANIFEST_PATH: Path = DATA_ROOT / "processed" / "manifest.json"

    # Reproducibility
    SEED: int = 42

    # Split ratios
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Corruption suite
    # None = corrupt the full test split; an int = sample that many images per class
    CORRUPTION_SAMPLE_PER_CLASS: int | None = None
    
    # Parallelism: None = use all CPU cores minus one
    NUM_WORKERS: int | None = 10


cfg = Config()