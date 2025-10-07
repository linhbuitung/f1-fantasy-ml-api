import joblib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.predict import save_metadata
from app.model import build_mainrace_pipeline  # if you have a helper; adapt as needed
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    path = Path("data/processed/cleaned_data_main_race_with_median.csv")
    if not path.exists():
        path = Path("data/processed/cleaned_data_median.csv")
    Xy = pd.read_csv(path)
    y = Xy["deviation_from_median"]
    X = Xy.drop(columns=["deviation_from_median"])
    pipeline = build_mainrace_pipeline()
    pipeline.fit(X, y)
    Path("model").mkdir(exist_ok=True)
    joblib.dump(pipeline, "model/trained_pipeline.pkl")
    # Save metadata: training time, dataset md5, sample params
    meta = {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target": "deviation_from_median",
    }
    save_metadata(meta, path=Path("model/metadata.json"))
    print("Model saved: model/trained_pipeline.pkl")