# ...existing code...
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, RootModel
from typing import List, Dict, Any
from app.schemas import MainRacePredictInput
from app.services.feature_builder import build_features_from_dto
from app.predict import load_model, predict_df

app = FastAPI()

class PredictRow(BaseModel):
    # allow arbitrary keys; validate in tests / caller
    pydantic.RootModel: Dict[str, Any]

class PredictRequest(BaseModel):
    rows: List[PredictRow]

@app.post("/predict")
def predict(req: MainRacePredictInput):
    dto = req.dict()
    features = build_features_from_dto(dto)  # -> dict of model features
    import pandas as pd
    df = pd.DataFrame([features])
    pipeline, meta = load_model()
    preds = predict_df(df, pipeline=pipeline)
    return {"predictions": preds.tolist(), "model_meta": meta}
