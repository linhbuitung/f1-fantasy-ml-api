# ...existing code...
from pathlib import Path
from typing import List

from fastapi import APIRouter
from app.schemas.dto import StatusPredictInput, StatusPredictionItem, StatusPredictResponse
from app.services.feature_builder import build_status_features_from_dto
from app.services.model_service import load_model, get_proba_df, get_batch_proba
import pandas as pd

router = APIRouter(prefix="/status", tags=["Predict Status"])

@router.get("/")
async def root():
    return {"Predict Status is running"}

@router.post("/predict", )
def predict(req: StatusPredictInput):
    input_dto = req.model_dump()

    # 2) expand minimal DTO into models features
    features = build_status_features_from_dto(input_dto)  # -> dict of models features

    # 3) predict
    df = pd.DataFrame([features])
    pipeline, meta = load_model(path="trained_status_pipeline.pkl", meta_path="status_metadata.json")
    predictions = get_proba_df(df, pipeline=pipeline)

    # 4) build response item(s)
    predicted_percentage = float(predictions.iloc[0])
    item = StatusPredictionItem(
        input=input_dto,
        features=features,
        dnf_percentage=predicted_percentage,
    )

    return StatusPredictResponse(percentages=[item], model_meta=meta)

@router.post("/predict/batch", response_model=StatusPredictResponse)
def predict_batch(reqs: List[StatusPredictInput]):
    """
    Accepts a list of minimal DTOs, expands each to model features,
    predicts batch deviations, and computes predicted_final_position per race.
    Returns the same QualifyingPredictResponse with one QualifyingPredictInput per input.
    """
    # 1) expand all DTOs to feature dicts
    inputs = [r.model_dump() for r in reqs]
    features_list = [build_status_features_from_dto(inp) for inp in inputs]

    # 2) build DataFrame in stable column order
    df = pd.DataFrame(features_list)

    # 3) predict + rank
    df_preds, meta = get_batch_proba(df, model_path="trained_status_pipeline.pkl",  meta_path="status_metadata.json")  # symbol: app.services.model_service.predict_batch_and_rank
    # 4) build response items preserving original inputs
    items = []
    for inp, feats, (_, row) in zip(inputs, features_list, df_preds.iterrows()):
        items.append(
            StatusPredictionItem(
                input=inp,
                features=feats,
                dnf_percentage=float(row["predicted_proba"]),
            )
        )

    return StatusPredictResponse(percentages=items, model_meta=meta)