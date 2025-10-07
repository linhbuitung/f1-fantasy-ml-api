from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

def serve_mainrace_df(
    raw_dir: Optional[str] = "data/raw",
    processed_dir: Optional[str] = "data/processed",
    weather_path: Optional[str] = None,
    date_col: str = "date",
    year_from: int = 1981,
) -> pd.DataFrame:
    """
    Build the merged DataFrame used as input to
    [`app.preprocess.create_training_datasets`](app/preprocess/preprocess_mainrace.py).

    - Loads raw CSVs from data/raw by default:
      data/raw/races.csv, results.csv, qualifying.csv, drivers.csv,
      constructors.csv, circuits.csv, status.csv, lap_times.csv
    - Performs the merges from the notebook:
      races -> results -> qualifying -> drivers -> constructors -> circuits -> status -> laptimes
    - Normalizes race dates, driver DOB, computes race_year/month/day,
      age_at_gp_in_days, first_race_date, days_since_first_race.
    - Merges weather (calls [`app.preprocess.process_raw_weather`](app/preprocess/preprocess_general.py) if weather_path not supplied)
    - Creates `rain` flag and returns the prepared DataFrame.
    """
    project_root = Path(__file__).resolve().parents[2]
    raw_base = Path(raw_dir) if raw_dir and Path(raw_dir).is_absolute() else (project_root / (raw_dir or "data" / "raw"))
    processed_base = Path(processed_dir) if processed_dir and Path(processed_dir).is_absolute() else (project_root / (processed_dir or "data" / "processed"))

    # load raw CSVs (fail early if missing)
    races = pd.read_csv(processed_base / "races.csv", dtype=str)
    results = pd.read_csv(raw_base / "results.csv", dtype=str)
    qualifyings = pd.read_csv(raw_base / "qualifying.csv", dtype=str)
    drivers = pd.read_csv(raw_base / "drivers.csv", dtype=str)
    constructors = pd.read_csv(raw_base / "constructors.csv", dtype=str)
    circuits = pd.read_csv(raw_base / "circuits.csv", dtype=str)
    status = pd.read_csv(raw_base / "status.csv", dtype=str)
    laptimes = pd.read_csv(raw_base / "lap_times.csv", dtype=str)

    countries = pd.read_csv(processed_base / "countries.csv", dtype=str)

    # Notebook merge order (same as Thesis.ipynb)
    df1 = pd.merge(races, results, how="left", on=["raceId"], suffixes=("_race", "_result"))
    df2 = pd.merge(df1, qualifyings, how="left", on=["raceId", "driverId", "constructorId"], suffixes=("", "_qualifying"))
    df3 = pd.merge(df2, drivers, how="left", on=["driverId"], suffixes=("", "_driver"))
    df4 = pd.merge(df3, constructors, how="left", on=["constructorId"], suffixes=("", "_constructor"))
    df5 = pd.merge(df4, circuits, how="left", on=["circuitId"], suffixes=("", "_circuit"))
    df6 = pd.merge(df5, status, how="left", on=["statusId"], suffixes=("", "_status"))
    df7 = pd.merge(df6, laptimes, how="left", on=["raceId", "driverId"], suffixes=("", "_laptime"))

    data = df7.copy()

    data = data.drop(['raceId', 'round', 'circuitId', 'name', 'time_race',
                      'url', 'fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date',
                      'fp3_time', 'quali_date', 'quali_time', 'sprint_date', 'sprint_time',
                      'resultId', 'constructorId', 'number', 'position',
                      'positionText', 'positionOrder', 'points', 'time_result',
                      'fastestLap', 'rank', 'fastestLapTime',
                      'fastestLapSpeed', 'statusId', 'qualifyId', 'number_qualifying',
                      'position_qualifying', 'q1', 'q2', 'q3', 'number_driver',
                      'code', 'url_driver',
                      'constructorRef',
                      'url_constructor', 'circuitRef', 'location',
                      'lat', 'lng', 'alt', 'url_circuit', 'lap', 'position_laptime', 'time'
                      ], axis=1)

    # rename/normalize columns used in notebook
    rename_map = {
        "name": "race_name",
        "grid": "qualification_position",
        "name_constructor": "constructor",
        "nationality": "driver_nationality",
        "nationality_constructor": "constructor_nationality",
        "name_circuit": "circuit",
        "country": "country_circuit",
        "type": "type_circuit",
        "dob": "driver_date_of_birth",
    }
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

    data['driver'] = data['driverRef']
    # drop forename columns and surname columns
    data = data.drop(['forename', 'surname', 'driverRef'], axis=1)

    # driver DOB -> datetime
    if "driver_date_of_birth" in data.columns:
        data["driver_date_of_birth"] = pd.to_datetime(data["driver_date_of_birth"], errors="coerce")

    # extract year/month/day
    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        data['race_month'] = data[date_col].dt.month
        data['race_day'] = data[date_col].dt.day
        data.rename(columns={'year': 'race_year'}, inplace=True)
        data['race_year'] = pd.to_numeric(data['race_year'], errors='coerce').astype(int)
        date = data.drop(date_col, axis=1)

    # compute age and first race fields (notebook logic)
    if "driver_date_of_birth" in data.columns and "date" in data.columns:
        data["age_at_gp_in_days"] = (data["date"] - data["driver_date_of_birth"]).abs().dt.days.fillna(0).astype("Int64", copy=False)

    if "driverId" in data.columns:
        first_race_dates = data.groupby("driverId")["date"].min().reset_index()
        first_race_dates = first_race_dates.rename(columns={"date": "first_race_date"})
        data = data.merge(first_race_dates, on="driverId", how="left")
        data["days_since_first_race"] = (data["date"] - data["first_race_date"]).abs().dt.days.fillna(0).astype("Int64", copy=False)

    data_tmp = data.copy()

    data_with_circuit_nationality = data_tmp.merge(
        countries,
        how='left',
        left_on='country_circuit',
        right_on='en_short_name'
    )

    # en_short_name

    data_with_circuit_nationality = data_with_circuit_nationality.drop(
        ['country_circuit', 'num_code', 'alpha_2_code', 'en_short_name', 'nationality'], axis=1)
    data_with_circuit_nationality.rename(columns={'alpha_3_code': 'circuit_nationality'}, inplace=True)

    data_tmp = data_with_circuit_nationality.copy()

    data_with_driver_nationality = data_tmp.merge(
        countries,
        how='left',
        left_on='driver_nationality',
        right_on='nationality'
    )

    data_with_driver_nationality = data_with_driver_nationality.drop(
        ['driver_nationality', 'num_code', 'alpha_2_code', 'en_short_name', 'nationality'], axis=1)
    data_with_driver_nationality.rename(columns={'alpha_3_code': 'driver_nationality'}, inplace=True)

    data_tmp = data_with_driver_nationality.copy()

    data_with_constructor_nationality = data_tmp.merge(
        countries,
        how='left',
        left_on='constructor_nationality',
        right_on='nationality'
    )
    data_with_constructor_nationality = data_with_constructor_nationality.drop(
        ['constructor_nationality', 'num_code', 'alpha_2_code', 'en_short_name', 'nationality'], axis=1)
    data_with_constructor_nationality.rename(columns={'alpha_3_code': 'constructor_nationality'}, inplace=True)

    # data after processing nationalities
    data = data_with_constructor_nationality;
    data['driver_home'] = data['driver_nationality'] == data['circuit_nationality']
    data['constructor_home'] = data['constructor_nationality'] == data['circuit_nationality']
    data['driver_home'] = data['driver_home'].apply(lambda x: int(x))
    data['constructor_home'] = data['constructor_home'].apply(lambda x: int(x))

    # take only date from year
    data = data[data['race_year'] >= year_from]

    # final housekeeping: drop exact duplicates and return
    data = data.drop_duplicates().reset_index(drop=True)
    return data

def create_training_datasets(
    df: pd.DataFrame,
    out_dir: str = "data/processed",
    min_laps_threshold: int = 10,
    deviation_lower: int = -110000,
    deviation_upper: int = 612000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Implements the preprocessing steps from research/Thesis.ipynb to produce:
      - data_median (detailed race rows with median and deviation)
      - cleaned_data_median (final cleaned dataset for training)

    Writes:
      - data_median.csv
      - data_median_race_duration.csv (median per circuit/year/date)
      - data_median.csv (detailed)
      - cleaned_data_median.csv (final features)

    Returns (data_median, cleaned_data_median)
    """
    project_root = Path(__file__).resolve().parents[2]
    out_dir_path = Path(out_dir) if Path(out_dir).is_absolute() else project_root / out_dir
    out_dir_path.mkdir(parents=True, exist_ok=True)

    data = df.copy()

    # 1) Group by all columns except 'milliseconds_laptime' and aggregate
    cols_to_group = [c for c in data.columns if c != "milliseconds_laptime"]
    grouped = (
        data.groupby(cols_to_group, as_index=False)
        .agg(milliseconds_laptime=("milliseconds_laptime", "sum"),
             laps_count=("milliseconds_laptime", "count"))
    )

    # 2) Numeric conversions and fill
    grouped["milliseconds_laptime"] = pd.to_numeric(grouped.get("milliseconds_laptime"), errors="coerce").fillna(0)
    grouped["milliseconds"] = pd.to_numeric(grouped.get("milliseconds"), errors="coerce").fillna(0)

    # 3) time_exist flag & filter
    grouped["time_exist"] = np.where(
        (grouped["milliseconds_laptime"].notna() & (grouped["milliseconds_laptime"] != 0))
        | (grouped["milliseconds"].notna() & (grouped["milliseconds"] != 0)),
        1,
        0,
    )
    grouped = grouped[grouped["time_exist"] != 0].copy()
    grouped.drop(columns=["time_exist"], inplace=True, errors="ignore")

    # 4) race_duration: prefer summed laptime else milliseconds
    grouped["race_duration"] = grouped.apply(
        lambda r: r["milliseconds_laptime"] if r["milliseconds_laptime"] != 0 else r["milliseconds"],
        axis=1,
    )

    # 5) laps adjustment: keep existing 'laps' but ensure >= laps_count
    if "laps" in grouped.columns:
        grouped["laps"] = grouped.apply(
            lambda r: r["laps"] if pd.notna(r["laps"]) and int(r["laps"]) >= int(r["laps_count"]) else int(r["laps_count"]),
            axis=1,
        )
    else:
        grouped["laps"] = grouped["laps_count"]

    grouped.drop(columns=["laps_count"], inplace=True, errors="ignore")

    # 6) compute max_laps per (circuit, race_year, date)
    grouped["laps"] = pd.to_numeric(grouped.get("laps"), errors="coerce")
    median_keys = [k for k in ("circuit", "race_year", "date") if k in grouped.columns]
    if median_keys:
        grouped["max_laps"] = grouped.groupby(median_keys)["laps"].transform("max")

    # 7) prepare data_median working frame
    data_median = grouped.copy()
    # drop duplicates (not harmful)
    data_median = data_median.drop_duplicates().reset_index(drop=True)
    # drop where laps < min_laps_threshold
    data_median = data_median[data_median["laps"] >= min_laps_threshold].copy()

    # 8) compute additional_laps and final_race_duration for finished or +N Lap statuses
    data_median["additional_laps"] = (data_median["max_laps"] - data_median["laps"]).abs().fillna(0)
    lap_pattern = r"\+.* Lap.*"
    finished_mask = data_median.get("status").eq("Finished") | data_median.get("status", "").astype(str).str.match(lap_pattern)
    # avoid division by zero
    data_median.loc[finished_mask & (data_median["laps"].fillna(0) != 0), "final_race_duration"] = (
        data_median["race_duration"]
        + data_median["race_duration"] / data_median["laps"] * data_median["additional_laps"]
    )
    data_median["final_race_duration"] = data_median["final_race_duration"].fillna(data_median["race_duration"])

    # 9) median per circuit/year/date and deviation
    if median_keys:
        med = (
            data_median.groupby(median_keys, as_index=False)
            .agg(median_race_duration=("final_race_duration", "median"))
        )
        data_median = data_median.merge(med, on=median_keys, how="left")
        data_median["deviation_from_median"] = data_median["final_race_duration"] - data_median["median_race_duration"]
    else:
        data_median["median_race_duration"] = np.nan
        data_median["deviation_from_median"] = np.nan

    # 10) filter statuses like notebook and drop small races
    data_median = data_median[
        data_median.get("status").eq("Finished") | data_median.get("status", "").astype(str).str.match(lap_pattern)
    ].copy()

    # 11) round numeric columns
    num_cols = data_median.select_dtypes(include=[np.number]).columns
    data_median[num_cols] = data_median[num_cols].round()

    # 12) filter extreme deviations
    data_median = data_median[
        (data_median["deviation_from_median"] > deviation_lower) & (data_median["deviation_from_median"] < deviation_upper)
    ].copy()

    # 13) final_position ranking per race for export (used later for evaluation)
    rank_keys = [k for k in ("race_year", "race_month", "race_day", "circuit") if k in data_median.columns]
    if rank_keys and "deviation_from_median" in data_median.columns:
        data_median["final_position"] = data_median.groupby(rank_keys)["deviation_from_median"].rank(method="min", ascending=True)

    # 14) create cleaned_data_median by dropping columns used only for computing metrics (mirror notebook)
    cols_to_drop = [
        "median_race_duration",
        "race_duration",
        "driver_date_of_birth",
        "first_race_date",
        "date",
        "milliseconds",
        "status",
        "milliseconds_laptime",
        "final_race_duration",
        "additional_laps",
        "laps",
        "final_position",
    ]
    cleaned = data_median.drop(columns=[c for c in cols_to_drop if c in data_median.columns], errors="ignore").copy()

    # rename columns to match notebook expectations
    if "max_laps" in cleaned.columns:
        cleaned = cleaned.rename(columns={"max_laps": "laps"})

    cleaned.to_csv(out_dir_path / "cleaned_data_main_race_with_median.csv", index=False)

    return data_median.reset_index(drop=True), cleaned.reset_index(drop=True)
