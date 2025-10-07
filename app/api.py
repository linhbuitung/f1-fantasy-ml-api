# ...existing code...
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from app.predict import load_model, predict_df

app = FastAPI()

class PredictRow(BaseModel):
    # allow arbitrary keys; validate in tests / caller
    __root__: Dict[str, Any]

class PredictRequest(BaseModel):
    rows: List[PredictRow]

@app.post("/predict")
def predict(req: PredictRequest):
    pipeline, meta = load_model()
    # convert list of dicts -> DataFrame
    rows = [r.__root__ for r in req.rows]
    import pandas as pd
    df = pd.DataFrame(rows)
    try:
        preds = predict_df(df, pipeline=pipeline)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"predictions": preds.tolist(), "model_meta": meta}
