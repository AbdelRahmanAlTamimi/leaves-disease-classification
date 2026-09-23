from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    DATA_DIR: Path = Path("../../data/raw/plantvillage/color")  
    SEED: int = 42
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15


cfg = Config()

