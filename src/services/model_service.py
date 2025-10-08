import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import pandas as pd

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
MAIN_RACE_MODEL_FILE = MODEL_DIR / "trained_mainrace_pipeline.pkl"
MAIN_RACE_META_FILE = MODEL_DIR / "mainrace_metadata.json"


def _git_commit_hash() -> Optional[str]:
    try:
        root = Path(__file__).resolve().parents[1]
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        return out
    except Exception:
        return None


def load_model(path: Optional[Path] = None):
    """
    Load models pipeline (joblib) and metadata. Returns (pipeline, metadata_dict).
    - path: optional path to .pkl file (overrides MODEL_FILE).
    """
    p = Path(path) if path else MAIN_RACE_MODEL_FILE
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    pipeline = joblib.load(p)
    meta = {}
    if MAIN_RACE_META_FILE.exists():
        try:
            meta = json.loads(MAIN_RACE_META_FILE.read_text())
        except Exception:
            meta = {}
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


def predict_record(record: Dict, pipeline=None, model_path: Optional[str] = None) -> float:
    """
    Predict for a single input record (dict -> DataFrame -> predict).
    """
    df = pd.DataFrame([record])
    return float(predict_df(df, pipeline=pipeline, model_path=model_path).iloc[0])


def save_metadata(extra: Dict, path: Optional[Path] = None):
    """
    Save metadata JSON next to models. Called by training script.
    """
    p = Path(path) if path else MAIN_RACE_META_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    base = {"git_commit": _git_commit_hash()}
    base.update(extra or {})
    p.write_text(json.dumps(base, indent=2))
