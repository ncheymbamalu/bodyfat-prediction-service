from dataclasses import dataclass
from pathlib import PosixPath

import pandas as pd


@dataclass
class Dataset:
    features: pd.DataFrame
    labels: pd.Series

    def write(self, features_path: PosixPath, labels_path: PosixPath) -> None:
        self.features.to_parquet(features_path)
        self.labels.to_frame().to_parquet(labels_path)

    @staticmethod
    def read(features_path: PosixPath, labels_path: PosixPath) -> "Dataset":
        feature_matrix: pd.DataFrame = pd.read_parquet(features_path)
        target_vector: pd.Series = pd.read_parquet(labels_path).squeeze()
        return Dataset(features=feature_matrix, labels=target_vector)
