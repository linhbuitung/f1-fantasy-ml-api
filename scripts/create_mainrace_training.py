import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.preprocess.preprocess_mainrace import serve_mainrace_df, create_mainrace_training_datasets
from pathlib import Path

if __name__ == "__main__":
    df = serve_mainrace_df()  # uses data/raw & data/processed defaults
    data_median, cleaned = create_mainrace_training_datasets(df=df)
    print("Wrote:", Path("data/processed").resolve())