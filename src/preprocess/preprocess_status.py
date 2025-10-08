from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional
import re

def serve_status_df(
    raw_dir: Optional[str] = "data/raw",
    processed_dir: Optional[str] = "data/processed",
    date_col: str = "date",
    year_from: int = 1981,
    ) -> pd.DataFrame:
    """
    Build the merged DataFrame used as input to
    [`src.preprocess.create_training_datasets`](src/preprocess/preprocess_qualifying.py).

    - Loads raw CSVs from data/raw by default:
      data/raw/races.csv, results.csv, qualifying.csv, drivers.csv,
      constructors.csv, circuits.csv, status.csv, lap_times.csv
    - Performs the merges from the notebook:
      races -> results -> qualifying -> drivers -> constructors -> circuits
    - Normalizes race dates, driver DOB, computes race_year/month/day,
      age_at_gp_in_days, first_race_date, days_since_first_race.
    """
    project_root = Path(__file__).resolve().parents[2]
    raw_base = Path(raw_dir) if raw_dir and Path(raw_dir).is_absolute() else (project_root / (raw_dir or "data" / "raw"))
    processed_base = Path(processed_dir) if processed_dir and Path(processed_dir).is_absolute() else (project_root / (processed_dir or "data" / "processed"))

    # load raw CSVs (fail early if missing)
    races = pd.read_csv(processed_base / "races.csv")
    results = pd.read_csv(raw_base / "results.csv")
    drivers = pd.read_csv(raw_base / "drivers.csv")
    constructors = pd.read_csv(raw_base / "constructors.csv")
    circuits = pd.read_csv(raw_base / "circuits.csv")
    status = pd.read_csv(raw_base / "status.csv")

    countries = pd.read_csv(processed_base / "countries.csv")

    # Notebook merge order (same as Thesis.ipynb)
    df1 = pd.merge(races, results, how="left", on=["raceId"], suffixes=("_race", "_result"))
    df2 = pd.merge(df1, drivers, how="left", on=["driverId"], suffixes=("", "_driver"))
    df3 = pd.merge(df2, constructors, how="left", on=["constructorId"], suffixes=("", "_constructor"))
    df4 = pd.merge(df3, circuits, how="left", on=["circuitId"], suffixes=("", "_circuit"))
    df5 = pd.merge(df4, status, how="left", on=["statusId"], suffixes=("", "_status"))

    data = df5.copy()

    data = data.drop(['raceId', 'round', 'circuitId', 'name', 'time_race', 'milliseconds',
                      'url', 'fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date',
                      'fp3_time', 'quali_date', 'quali_time', 'sprint_date', 'sprint_time',
                      'resultId', 'constructorId', 'number', 'position',
                      'positionText', 'positionOrder', 'points', 'time_result',
                      'fastestLap', 'rank', 'fastestLapTime',
                      'fastestLapSpeed', 'statusId', 'number_driver',
                      'code', 'url_driver', 'name_circuit', 'name_constructor',
                      'url_constructor', 'location', 'lat', 'lng', 'alt', 'url_circuit',
                      'forename', 'surname', 'code'
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
        'circuitRef': 'circuit',
        'constructorRef': 'constructor',
        'driverRef': 'driver',
    }
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

    # replace certain nationalities to match countries.csv
    data['constructor_nationality'] = data['constructor_nationality'].str.strip()
    data['constructor_nationality'] = data['constructor_nationality'].replace(
        {'Rhodesian': 'Zimbabwean',
        'American-Italian': 'American',
         'Argentine-Italian': 'Argentine',
         'East German': 'German',
         'West German': 'German',
         'Argentinian' : 'Argentine',})

    data['driver_nationality'] = data['driver_nationality'].str.strip()
    data['driver_nationality'] = data['driver_nationality'].replace(
        {'Rhodesian': 'Zimbabwean',
        'American-Italian': 'American',
         'Argentine-Italian': 'Argentine',
         'East German': 'German',
         'West German': 'German',
         'Argentinian' : 'Argentine',})


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

    data = data[data['race_year'] >= year_from]
    # final housekeeping: drop exact duplicates and return
    data = data.drop_duplicates().reset_index(drop=True)
    return data

def create_status_training_datasets(
    df: pd.DataFrame,
    out_dir: str = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Port of the Thesis-Status notebook flow
    Returns (data_status, cleaned)
    """
    project_root = Path(__file__).resolve().parents[2]
    out_dir_path = Path(out_dir) if Path(out_dir).is_absolute() else project_root / out_dir
    out_dir_path.mkdir(parents=True, exist_ok=True)

    data_cleaned_status = df.copy()

    # drop duplicates
    data_cleaned_status.drop_duplicates(inplace=True)

    pattern = r"\+.* Lap.*"
    data_cleaned_status['dnf'] = data_cleaned_status['status'].apply(lambda x: 0 if x == 'Finished' or re.match(pattern, x) else 1)


    # prepare cleaned (drop columns used for metrics, keep parity with other create_* funcs)
    cols_to_drop = [
        "driver_date_of_birth",
        "first_race_date",
        "date",
        "laps",
        "status",
        "driverId",
        "constructorId",
        "final_position"
        # keep domain specific columns as needed
    ]
    cleaned = data_cleaned_status.drop(columns=[c for c in cols_to_drop if c in data_cleaned_status.columns], errors="ignore").copy()

    cleaned.to_csv(out_dir_path / "cleaned_data_status.csv", index=False)

    return data_cleaned_status.reset_index(drop=True), cleaned.reset_index(drop=True)