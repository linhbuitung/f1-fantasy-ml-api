from pathlib import Path
import pandas as pd
from src.services.model_service import load_model, predict_df, get_proba_df

def test_mainrace_model_load_and_predict_smoke():
    # CI should dvc pull models/trained_pipeline.pkl, or run a tiny local test models
    model_path = Path("models/trained_mainrace_pipeline.pkl")
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
        "driver": "uvicorn main:app --log-level trace",
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

def test_qualifying_model_load_and_predict_smoke():
    # CI should dvc pull models/trained_qualifying_pipeline.pkl or train locally
    model_path = Path("models/trained_qualifying_pipeline.pkl")
    assert model_path.exists(), "Model file missing (run dvc pull or train)"
    pipeline, meta = load_model(path=model_path)

    df = pd.DataFrame([{
        "age_at_gp_in_days": 12000,
        "days_since_first_race": 1000,
        "constructor": "ferrari",
        "circuit": "yas_marina",
        "type_circuit": "Race circuit",
        "driver": "test_driver",
        "circuit_nationality": "ARE",
        "driver_nationality": "GBR",
        "constructor_nationality": "ITA",
        "race_year": 2021,
        "race_month": 12,
        "race_day": 12,
        "driver_home": 0,
        "constructor_home": 0,
    }])

    preds = predict_df(df, pipeline=pipeline)
    assert len(preds) == 1

def test_status_model_load_and_proba_smoke():
    # CI should dvc pull models/trained_status_pipeline.pkl or train locally
    model_path = Path("models/trained_status_pipeline.pkl")
    assert model_path.exists(), "Model file missing (run dvc pull or train)"
    pipeline, meta = load_model(path=model_path)

    df = pd.DataFrame([{
        "qualification_position": 12,
        "age_at_gp_in_days": 10000,
        "days_since_first_race": 800,
        "constructor": "mercedes",
        "circuit": "silverstone",
        "type_circuit": "Race circuit",
        "driver": "test_driver",
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

    probs = get_proba_df(df, pipeline=pipeline)
    assert len(probs) == 1
    assert 0.0 <= float(probs.iloc[0]) <= 100.0