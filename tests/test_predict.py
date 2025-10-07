from pathlib import Path
import pandas as pd
from app.predict import load_model, predict_df

def test_model_load_and_predict_smoke():
    # CI should dvc pull model/trained_pipeline.pkl, or run a tiny local test model
    model_path = Path("model/trained_pipeline.pkl")
    assert model_path.exists(), "Model file missing (run dvc pull or train)"
    pipeline, meta = load_model()
    # construct a minimal DF using columns mentioned in research/Thesis.ipynb defaults
    df = pd.DataFrame([{
        "qualification_position": 10,
        "age_at_gp_in_days": 12000,
        "days_since_first_race": 1000,
        "laps": 58,
        "constructor": "ferrari",
        "circuit": "silverstone",
        "type_circuit": "Race circuit",
        "driver": "hamilton",
        "circuit_nationality": "GBR",
        "driver_nationality": "GBR",
        "constructor_nationality": "GBR",
        "race_year": 2023,
        "race_month": 7,
        "race_day": 14,
        "rain": 0,
        "driver_home": 1,
        "constructor_home": 0,
    }])
    preds = predict_df(df, pipeline=pipeline)
    assert len(preds) == 1
