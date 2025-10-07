import joblib
from app.model import build_pipeline  # if you have a helper; adapt as needed
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    Xy = pd.read_csv("data/processed/cleaned_data_median.csv")
    y = Xy["deviation_from_median"]
    X = Xy.drop(columns=["deviation_from_median"])
    pipeline = build_pipeline()  # implement in app/model.py to return sklearn Pipeline
    pipeline.fit(X, y)
    Path("model").mkdir(exist_ok=True)
    joblib.dump(pipeline, "model/trained_pipeline.pkl")
    print("Model saved: model/trained_pipeline.pkl")