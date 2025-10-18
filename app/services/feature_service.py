from pathlib import Path

import pandas as pd
from fastapi import HTTPException

project_root = Path(__file__).resolve().parents[2]
FEATURES_HELPER_DIR = project_root / "data" / "processed" / "features_helper"


def read_options_csv(primary_name: str, fallback_name: str):
    """
    Attempt to read primary CSV under features_helper (e.g. drivers_mainrace.csv),
    otherwise fall back to processed/{fallback_name} (e.g. drivers.csv).
    Returns list[dict] (records).
    """
    primary = FEATURES_HELPER_DIR / primary_name
    if primary.exists():
        try:
            df = pd.read_csv(primary)
            return df.fillna("").to_dict(orient="records")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed reading {primary}: {e}")

    fallback = project_root / "data" / "processed" / fallback_name
    if fallback.exists():
        try:
            df = pd.read_csv(fallback)
            return df.fillna("").to_dict(orient="records")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed reading {fallback}: {e}")

    raise HTTPException(status_code=404, detail=f"No options found (tried {primary} and {fallback})")