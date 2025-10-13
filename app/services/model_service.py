import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[2]  # repo root (.. / .. from this file)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(project_root / "models")))
MAIN_RACE_MODEL_FILE = MODEL_DIR / "trained_mainrace_pipeline.pkl"
MAIN_RACE_META_FILE = MODEL_DIR / "mainrace_metadata.json"
QUALIFYING_MODEL_FILE = MODEL_DIR / "trained_qualifying_pipeline.pkl"
QUALIFYING_META_FILE = MODEL_DIR / "qualifying_metadata.json"

def _git_commit_hash() -> Optional[str]:
    try:
        root = Path(__file__).resolve().parents[1]
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        return out
    except Exception:
        return None


def load_model(path: Optional[str] = None, meta_path: Optional[str] = None) -> tuple:
    """
    Load models pipeline (joblib) and metadata. Returns (pipeline, metadata_dict).
    - path: optional path to .pkl file (overrides MODEL_FILE).
    """
    p = MODEL_DIR / path
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    pipeline = joblib.load(p)

    p_meta = MODEL_DIR / meta_path
    if not p_meta.exists():
        raise FileNotFoundError(f"Meta not found: {p}")
    meta = json.loads(p_meta.read_text())
    # attach runtime provenance if missing
    meta.setdefault("git_commit", _git_commit_hash())
    return pipeline, meta


def predict_df(df: pd.DataFrame, pipeline=None, model_path: Optional[str] = None) -> pd.Series:
    """
    Predict on a DataFrame. Accepts raw feature columns as expected by the
    training pipeline (same names/order isn't required if ColumnTransformer
    + OneHotEncoder with handle_unknown='ignore' used).
    Returns a pd.Series of predictions.
    """
    if pipeline is None:
        pipeline, _ = load_model(model_path)
    preds = pipeline.predict(df)
    return pd.Series(preds, index=df.index)

def get_proba_df(df: pd.DataFrame, pipeline=None, model_path: Optional[str] = None) -> pd.Series:
    """
    Get probability on a DataFrame. Accepts raw feature columns as expected by the
    training pipeline (same names/order isn't required if ColumnTransformer
    + OneHotEncoder with handle_unknown='ignore' used).
    Returns a pd.Series of probabilities for the positive class.
    """
    if pipeline is None:
        pipeline, _ = load_model(model_path)
    preds = pipeline.predict_proba(df)[:, 1]*100  # probability of positive class
    return pd.Series(preds, index=df.index)

def predict_record(record: Dict, pipeline=None, model_path: Optional[str] = None) -> float:
    """
    Predict for a single input record (dict -> DataFrame -> predict).
    """
    df = pd.DataFrame([record])
    return float(predict_df(df, pipeline=pipeline, model_path=model_path).iloc[0])

def get_record_percentiles_for_classification(record: Dict, pipeline=None, model_path: Optional[str] = None) -> float:
    """
    Get percentage of classification for a single input record (dict -> DataFrame -> predict).
    """
    df = pd.DataFrame([record])
    return float(get_proba_df(df, pipeline=pipeline, model_path=model_path).iloc[0])


def save_metadata(extra: Dict, path: Optional[Path] = None):
    """
    Save metadata JSON next to models. Called by training script.
    """
    p = Path(path) if path else MAIN_RACE_META_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    base = {"git_commit": _git_commit_hash()}
    base.update(extra or {})
    p.write_text(json.dumps(base, indent=2))

def predict_batch_and_rank(
    df: pd.DataFrame,
    pipeline=None,
    model_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    rank_keys: Optional[list] = None,
) -> tuple[pd.DataFrame, Dict]:
    """
    Predict for a batch DataFrame and compute predicted_final_position per race.
    - df: DataFrame of expanded features (must include rank_keys columns)
    - pipeline: optional preloaded pipeline
    - model_path: optional model path (resolved by load_model)
    - rank_keys: list of columns to define a race group, defaults to ["race_year","race_month","race_day","circuit"]
    Returns (df_out, meta) where df_out contains 'predicted_deviation_from_median' and 'predicted_final_position'.
    """
    if pipeline is None:
        pipeline, meta = load_model(path= model_path, meta_path=meta_path)
    else:
        _, meta = load_model(path=model_path, meta_path=meta_path) if model_path else ({}, {})  # ensure meta loaded when requested
    # predict
    predictions = pipeline.predict(df)
    df_out = df.copy()
    df_out["predicted_deviation_from_median"] = np.asarray(predictions).astype(float)

    # determine grouping keys for ranking
    if rank_keys is None:
        rank_keys = ["race_year", "race_month", "race_day", "circuit"]
    # keep only keys that exist
    used_keys = [k for k in rank_keys if k in df_out.columns]
    if not used_keys:
        # fallback: rank globally
        df_out["predicted_final_position"] = df_out["predicted_deviation_from_median"].rank(
            method="min", ascending=True
        ).astype(int)
        return df_out, meta

    # compute predicted final position per group (lower deviation -> better pos)
    df_out["predicted_final_position"] = (
        df_out.groupby(used_keys)["predicted_deviation_from_median"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    return df_out, meta

def get_batch_proba(
    df: pd.DataFrame,
    pipeline=None,
    model_path: Optional[str] = None,
    meta_path: Optional[str] = None,
) -> tuple[pd.DataFrame, Dict]:
    """
    Predict probabilities for a batch DataFrame.
    - df: DataFrame of expanded features (must include rank_keys columns)
    - pipeline: optional preloaded pipeline
    - model_path: optional model path (resolved by load_model)
    Returns (df_out, meta) where df_out contains 'predicted_proba' and.
    """
    if pipeline is None:
        pipeline, meta = load_model(path=model_path, meta_path=meta_path)
    else:
        _, meta = load_model(path=model_path, meta_path=meta_path) if model_path else ({}, {})
    # predict probabilities
    predictions = pipeline.predict_proba(df)[:, 1]*100
    df_out = df.copy()
    df_out["predicted_proba"] = np.asarray(predictions).astype(float)

    return df_out, meta