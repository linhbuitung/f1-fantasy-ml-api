import joblib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.model_service import save_metadata
from app.models.status_pipeline import build_status_pipeline  # if you have a helper; adapt as needed
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    path = ROOT / "data" / "processed" / "cleaned_data_status.csv"

    Xy = pd.read_csv(path)
    y = Xy["dnf"]
    X = Xy.drop(columns=["dnf"])
    pipeline = build_status_pipeline()
    pipeline.fit(X, y)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipeline, "models/trained_status_pipeline.pkl")
    # Save metadata: training time, dataset md5, sample params
    meta = {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target": "dnf",
    }
    save_metadata(meta, path=Path("models/status_metadata.json"))
    print("Model saved: models/trained_status_pipeline.pkl")