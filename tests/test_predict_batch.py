import pandas as pd
import numpy as np

from app.services.model_service import predict_batch_and_rank

class FakePipeline:
    """Deterministic fake pipeline: returns qualification_position as prediction if present."""
    def predict(self, X):
        if "qualification_position" in X.columns:
            return X["qualification_position"].to_numpy().astype(float)
        return np.zeros(len(X))

def test_predict_batch_and_rank_grouping():
    # two races (groups) with multiple drivers
    rows = [
        # race A
        {"race_year": 2023, "race_month": 7, "race_day": 14, "circuit": "silverstone", "qualification_position": 10},
        {"race_year": 2023, "race_month": 7, "race_day": 14, "circuit": "silverstone", "qualification_position": 5},
        {"race_year": 2023, "race_month": 7, "race_day": 14, "circuit": "silverstone", "qualification_position": 1},
        # race B
        {"race_year": 2023, "race_month": 8, "race_day": 20, "circuit": "yas_marina", "qualification_position": 2},
        {"race_year": 2023, "race_month": 8, "race_day": 20, "circuit": "yas_marina", "qualification_position": 4},
    ]
    df = pd.DataFrame(rows)
    pipeline = FakePipeline()

    df_out, meta = predict_batch_and_rank(df, pipeline=pipeline)

    # predictions equal qualification_position
    assert np.allclose(df_out["predicted_deviation_from_median"].to_numpy(), df["qualification_position"].to_numpy().astype(float))

    # expected ranks per group (ascending prediction -> better position)
    # race A: qpos [10,5,1] -> ranks [3,2,1]
    # race B: qpos [2,4] -> ranks [1,2]
    expected_ranks = [3,2,1,1,2]
    assert df_out["predicted_final_position"].tolist() == expected_ranks

def test_predict_batch_and_rank_global_fallback():
    # no grouping keys present -> fallback to global ranking
    df = pd.DataFrame([
        {"qualification_position": 3},
        {"qualification_position": 1},
        {"qualification_position": 2},
    ])
    pipeline = FakePipeline()
    df_out, meta = predict_batch_and_rank(df, pipeline=pipeline)

    # predictions
    assert np.allclose(df_out["predicted_deviation_from_median"].to_numpy(), np.array([3.0,1.0,2.0]))

    # global ranks ascending -> [3,1,2]
    assert df_out["predicted_final_position"].tolist() == [3,1,2]