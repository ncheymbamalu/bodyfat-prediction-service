import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field, NonNegativeFloat
from xgboost import XGBRegressor

from src.config import Config


model: XGBRegressor = XGBRegressor()
model.load_model(Config.Path.MODELS_DIR / "model.json")

defaults: dict[str, float] = (
    round(
        pd.read_parquet(Config.Path.DATA_DIR / "processed" / "X_train.parquet")
        .drop("sex", axis=1)
        .mean(),
        2
    )
    .to_dict()
)

class Record(BaseModel):
    weight: NonNegativeFloat = Field(default=defaults.get("weight"), alias="Weight (kg)")
    height: NonNegativeFloat = Field(default=defaults.get("height"), alias="Height (kg)")
    chest: NonNegativeFloat = Field(default=defaults.get("chest"), alias="Chest (cm)")
    abdomen: NonNegativeFloat = Field(default=defaults.get("abdomen"), alias="Abdomen (cm)")
    hip: NonNegativeFloat = Field(default=defaults.get("hip"), alias="Hip (cm)")
    biceps: NonNegativeFloat = Field(default=defaults.get("biceps"), alias="Biceps (cm)")


app: FastAPI = FastAPI(
    title="Bodyfat Prediction API", 
    description="REST API to predict bodyfat percentage based on personal measurements"
)

@app.post("/predict", response_model=dict[str, float])
def make_prediction(input_data: Record):
    record: pd.DataFrame = pd.DataFrame([input_data.model_dump()])
    record = (
        record
        .assign(
            bmi=record["weight"] / (record["height"] ** 2), 
            bai=(record["hip"] / (record["height"] ** 1.5)) - 18, 
            whr=record["abdomen"] / record["hip"]
        )
        [model.feature_names_in_]
    )
    prediction: float = model.predict(record)[0]
    return {"prediction": prediction}
