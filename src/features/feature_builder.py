from pathlib import PosixPath

import pandas as pd

from omegaconf import ListConfig

from src.config import Config, load_config


class FeatureBuilder:
    def __init__(self, train_features_path: PosixPath, test_features_path: PosixPath):
        self.train_features_path: PosixPath = train_features_path
        self.test_features_path: PosixPath = test_features_path
        self.features: ListConfig = load_config().features.mutual_info

    def engineer_features(self, path: PosixPath) -> pd.DataFrame:
        data: pd.DataFrame = pd.read_parquet(path)
        return (
            data
            .assign(
                bmi=data["weight"] / (data["height"] ** 2), 
                bai=(data["hip"] / (data["height"] ** 1.5)) - 18, 
                whr=data["abdomen"] / data["hip"]
            )
            [self.features]
        )
    
    def build(self, subsets: list[str] = ["train", "test"]) -> None:
        Config.Path.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        for path, subset in zip([self.train_features_path, self.test_features_path], subsets):
            self.engineer_features(path).to_parquet(Config.Path.FEATURES_DIR / f"{subset}.parquet")
