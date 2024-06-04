import numpy as np
import pandas as pd

from dvclive import Live
from omegaconf import DictConfig
from xgboost import XGBRegressor

from src.config import Config, load_config
from src.data.dataset import Dataset

CONFIG: DictConfig = load_config()


def get_rsquared(y: np.ndarray | pd.Series, yhat: np.ndarray | pd.Series) -> float:
    """Computes the R² between y and yhat

    Args:
        y (np.ndarray | pd.Series): Labels
        yhat (np.ndarray | pd.Series): Predictions

    Returns:
        float: R²
    """
    t: np.ndarray | pd.Series = y - y.mean()
    sst: float = t.dot(t)
    e: np.ndarray | pd.Series = y - yhat
    sse: float = e.dot(e)
    return 1 - (sse / sst)


def evaluate(model: XGBRegressor, dataset: Dataset) -> float:
    """Generates predictions and returns the corresponding R²

    Args:
        model (XGBRegressor): Regressor
        dataset (Dataset): Object that stores the features and labels

    Returns:
        float: R²
    """
    predictions: np.ndarray = model.predict(dataset.features)
    return round(get_rsquared(dataset.labels, predictions), 4)


def train_model(train_set: Dataset, val_set: Dataset) -> None:
    """Trains an object of type, 'XGBRegressor', logs its corresponding 
    train and validation set metrics, and writes the trained 'XGBRegressor' 
    object to './artifacts/models/model.json'

    Args:
        train_set (Dataset): Object that stores the train set features and labels
        val_set (Dataset): Object that stores the validation set features and labels
    """
    for dir_path in [Config.Path.EXPERIMENTS_DIR, Config.Path.MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    with Live(Config.Path.EXPERIMENTS_DIR) as live:
        model: XGBRegressor = XGBRegressor(
            n_estimators=CONFIG.model.n_estimators, max_depth=CONFIG.model.max_depth
        )
        model.fit(train_set.features, train_set.labels)
        train_metric: float = evaluate(model, train_set)
        live.log_metric("Train R²", train_metric)
        val_metric: float = evaluate(model, val_set)
        live.log_metric("Validation R²", val_metric)
        model.save_model(Config.Path.MODELS_DIR / "model.json")
