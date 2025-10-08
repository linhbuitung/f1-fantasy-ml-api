# ...existing code...
from fastapi import APIRouter
from src.schemas.dto import MainRacePredictInput, PredictionItem, MainRacePredictResponse
from src.services.feature_builder import build_features_from_dto
from src.services.model_service import load_model, predict_df
import pandas as pd

router = APIRouter(prefix="/main-race", tags=["Predict Main Race"])

@router.get("/")
async def root():
    return {"App is running"}

@router.post("/predict", response_model=MainRacePredictResponse)
def predict(req: MainRacePredictInput):
    input_dto = req.model_dump()

    # 2) expand minimal DTO into models features
    features = build_features_from_dto(input_dto)  # -> dict of models features

    # 3) predict
    df = pd.DataFrame([features])
    pipeline, meta = load_model("trained_mainrace_pipeline.pkl")
    predictions = predict_df(df, pipeline=pipeline)

    # 4) build response item(s)
    predicted_deviation = float(predictions.iloc[0])
    item = PredictionItem(
        input=input_dto,
        features=features,
        predicted_deviation_from_median=predicted_deviation,
        predicted_final_position=None,  # compute if you provide full race batch
    )

    return MainRacePredictResponse(predictions=[item], model_meta=meta)

