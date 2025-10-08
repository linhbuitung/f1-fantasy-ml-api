import joblib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.model_service import save_metadata
from src.models.qualifying_pipeline import build_qualifying_pipeline  # if you have a helper; adapt as needed
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    path = Path("data/processed/cleaned_data_qualifying_with_median.csv")

    Xy = pd.read_csv(path)
    y = Xy["deviation_from_median"]
    X = Xy.drop(columns=["deviation_from_median"])
    pipeline = build_qualifying_pipeline()
    pipeline.fit(X, y)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipeline, "models/trained_qualifying_pipeline.pkl")
    # Save metadata: training time, dataset md5, sample params
    meta = {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target": "deviation_from_median",
    }
    save_metadata(meta, path=Path("models/qualifying_metadata.json"))
    print("Model saved: models/trained_qualifying_pipeline.pkl")