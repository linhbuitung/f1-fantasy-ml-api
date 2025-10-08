import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.preprocess_general import *

if __name__ == "__main__":
    print("Rebuilding processed data...")
    build_all_general_processed_data()
    print("Done.")