import random

from pathlib import Path, PosixPath

import numpy as np

from omegaconf import DictConfig, OmegaConf


class Config:
    class Path:
        APP_HOME: PosixPath = Path(__file__).parent.parent
        EXPERIMENTS_DIR: PosixPath = APP_HOME / "experiments"
        ARTIFACTS_DIR: PosixPath = APP_HOME / "artifacts"
        DATA_DIR: PosixPath = ARTIFACTS_DIR / "data"
        FEATURES_DIR: PosixPath = ARTIFACTS_DIR / "features"
        MODELS_DIR: PosixPath = ARTIFACTS_DIR / "models"


def load_config(path: PosixPath = Config.Path.APP_HOME / "params.yaml") -> DictConfig:
    return OmegaConf.load(path)


def seed_everything(seed: int = load_config().random_seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
