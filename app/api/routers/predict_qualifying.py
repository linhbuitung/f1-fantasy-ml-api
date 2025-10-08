# ...existing code...
from pathlib import Path
from typing import List

from fastapi import APIRouter
from app.schemas.dto import QualifyingPredictInput, QualifyingPredictionItem, QualifyingPredictResponse
from app.services.feature_builder import build_qualifying_features_from_dto
from app.services.model_service import load_model, predict_df, predict_batch_and_rank
import pandas as pd

router = APIRouter(prefix="/qualifying", tags=["Predict Qualifying"])

@router.get("/")
async def root():
    return {"Predict Qualifying is running"}

@router.post("/predict", response_model=QualifyingPredictResponse)
def predict(req: QualifyingPredictInput):
    input_dto = req.model_dump()

    # 2) expand minimal DTO into models features
    features = build_qualifying_features_from_dto(input_dto)  # -> dict of models features

    # 3) predict
    df = pd.DataFrame([features])
    pipeline, meta = load_model(path=Path("models/trained_qualifying_pipeline.pkl"))
    predictions = predict_df(df, pipeline=pipeline)

    # 4) build response item(s)
    predicted_deviation = float(predictions.iloc[0])
    item = QualifyingPredictionItem(
        input=input_dto,
        features=features,
        predicted_deviation_from_median=predicted_deviation,
        predicted_final_position=None,  # compute if you provide full race batch
    )

    return QualifyingPredictResponse(predictions=[item], model_meta=meta)

@router.post("/predict/batch", response_model=QualifyingPredictResponse)
def predict_batch(reqs: List[QualifyingPredictInput]):
    """
    Accepts a list of minimal DTOs, expands each to model features,
    predicts batch deviations, and computes predicted_final_position per race.
    Returns the same QualifyingPredictResponse with one QualifyingPredictInput per input.
    """
    # 1) expand all DTOs to feature dicts
    inputs = [r.model_dump() for r in reqs]
    features_list = [build_qualifying_features_from_dto(inp) for inp in inputs]

    # 2) build DataFrame in stable column order
    df = pd.DataFrame(features_list)

    # 3) predict + rank
    df_preds, meta = predict_batch_and_rank(df, model_path="models/trained_qualifying_pipeline.pkl")  # symbol: app.services.model_service.predict_batch_and_rank
    # 4) build response items preserving original inputs
    items = []
    for inp, feats, (_, row) in zip(inputs, features_list, df_preds.iterrows()):
        items.append(
            QualifyingPredictionItem(
                input=inp,
                features=feats,
                predicted_deviation_from_median=float(row["predicted_deviation_from_median"]),
                predicted_final_position=int(row["predicted_final_position"]),
            )
        )

    return QualifyingPredictResponse(predictions=items, model_meta=meta)