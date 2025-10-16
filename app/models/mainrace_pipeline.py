from typing import Iterable, Optional, Sequence
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


def build_mainrace_pipeline(
    numeric_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    passthrough_cols: Optional[Sequence[str]] = None,
    estimator: str = "gbr",
):

    # sensible defaults matching research/Thesis.ipynb
    if numeric_cols is None:
        numeric_cols = ["qualification_position", "age_at_gp_in_days", "days_since_first_race", "laps", "race_year"]
    if categorical_cols is None:
        categorical_cols = [
            "constructor",
            "circuit",
            "type_circuit",
            "driver",
            "circuit_nationality",
            "driver_nationality",
            "constructor_nationality",
            "race_month",
            "race_day",
            "rain",
            "driver_home",
            "constructor_home"
        ]
    if passthrough_cols is None:
        passthrough_cols = []

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), list(numeric_cols)),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical_cols)),
        ],
        remainder="passthrough",  # passthrough_cols will remain, but ColumnTransformer doesn't accept explicit passthrough list; caller should ensure ordering if needed
    )

    est_lower = (estimator or "gbr").lower()
    if est_lower == "gbr":
        model = GradientBoostingRegressor(
            learning_rate=0.1,
            random_state=42,
            loss='huber',
            max_depth=5,
            min_samples_leaf=4,
            min_samples_split=5,
            n_estimators=400)
    else:
        # fallback
        model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            max_depth=30,
            min_samples_leaf=2,
            min_samples_split=10,
            n_estimators=800)

    pipeline = Pipeline([("preprocessing", preprocessor), ("models", model)])
    return pipeline