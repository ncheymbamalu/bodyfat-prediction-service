import sqlite3

from pathlib import PosixPath

import pandas as pd

from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.config import Config, load_config


class DatasetBuilder:
    def __init__(self, database_path: PosixPath):
        self.database_path: PosixPath = database_path
        self.processed_dir: PosixPath = Config.Path.DATA_DIR / "processed"
        self.config: DictConfig = load_config()


    def ingest_data(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM bodyfat", sqlite3.connect(self.database_path))
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data: pd.DataFrame = data.drop("Original", axis=1)
        data.columns = self.config.data.column_names
        return data
    
    def save_subset(self, features: pd.DataFrame, labels: pd.Series, subset: str) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        features.to_parquet(self.processed_dir / f"X_{subset}.parquet")
        labels.to_frame().to_parquet(self.processed_dir / f"y_{subset}.parquet")
    
    def split_and_save(self, data: pd.DataFrame, target_name: str = "body_fat") -> None:
        x_train, x_test, y_train, y_test = train_test_split(
            data.drop(target_name, axis=1), 
            data[target_name], 
            test_size=self.config.data.test_size, 
            stratify=data["sex"]
        )
        self.save_subset(x_train, y_train, "train")
        self.save_subset(x_test, y_test, "test")
    
    def build(self) -> None:
        self.ingest_data().pipe(self.preprocess_data).pipe(self.split_and_save)
