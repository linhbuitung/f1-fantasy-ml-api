from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

def serve_qualifying_df(
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
    qualifyings = pd.read_csv(raw_base / "qualifying.csv")
    drivers = pd.read_csv(raw_base / "drivers.csv")
    constructors = pd.read_csv(raw_base / "constructors.csv")
    circuits = pd.read_csv(raw_base / "circuits.csv")

    countries = pd.read_csv(processed_base / "countries.csv")

    # Notebook merge order (same as Thesis.ipynb)
    df1 = pd.merge(races, results, how="left", on=["raceId"], suffixes=("_race", "_result"))
    df2 = pd.merge(df1, qualifyings, how="left", on=["raceId", "driverId", "constructorId"], suffixes=("", "_qualifying"))
    df3 = pd.merge(df2, drivers, how="left", on=["driverId"], suffixes=("", "_driver"))
    df4 = pd.merge(df3, constructors, how="left", on=["constructorId"], suffixes=("", "_constructor"))
    df5 = pd.merge(df4, circuits, how="left", on=["circuitId"], suffixes=("", "_circuit"))

    data = df5.copy()

    data = data.drop(['raceId', 'round', 'circuitId', 'name', 'time_race', 'milliseconds',
                      'url', 'fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date',
                      'fp3_time', 'quali_date', 'quali_time', 'sprint_date', 'sprint_time', 'rain',
                      'resultId', 'constructorId', 'grid', 'number', 'position',
                      'positionText', 'positionOrder', 'points', 'laps', 'time_result',
                      'fastestLap', 'rank', 'fastestLapTime',
                      'fastestLapSpeed', 'statusId', 'qualifyId', 'number_qualifying',
                      'position_qualifying', 'number_driver',
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

    data = data[data['race_year'] >= year_from]
    # final housekeeping: drop exact duplicates and return
    data = data.drop_duplicates().reset_index(drop=True)
    return data

def create_qualifying_training_datasets(
    df: pd.DataFrame,
    out_dir: str = "data/processed"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Port of the Thesis-Qualifying notebook flow:
      - clean q1/q2/q3 strings, convert to timedelta -> milliseconds
      - compute milliseconds_qualification (min of q1,q2,q3)
      - drop zeros/duplicates
      - compute median per (circuit, race_year, date)
      - compute deviation_from_median and final_position (group rank)
      - round numeric columns and write CSV to out_dir_path
    Returns (data_median, cleaned)
    """
    project_root = Path(__file__).resolve().parents[2]
    out_dir_path = Path(out_dir) if Path(out_dir).is_absolute() else project_root / out_dir
    out_dir_path.mkdir(parents=True, exist_ok=True)

    data_cleaned_quali_time = df.copy()

    # Replace invalid values with a default time (e.g., '0:00.000')
    data_cleaned_quali_time['q1'] = data_cleaned_quali_time['q1'].fillna('0:00.000').replace('/N', '0:00.000')
    data_cleaned_quali_time['q2'] = data_cleaned_quali_time['q2'].fillna('0:00.000').replace('/N', '0:00.000')
    data_cleaned_quali_time['q3'] = data_cleaned_quali_time['q3'].fillna('0:00.000').replace('/N', '0:00.000')

    # Prepend '00:' to convert to hh:mm:ss format
    data_cleaned_quali_time['q1'] = pd.to_timedelta('00:' + data_cleaned_quali_time['q1'], errors='coerce')
    data_cleaned_quali_time['q2'] = pd.to_timedelta('00:' + data_cleaned_quali_time['q2'], errors='coerce')
    data_cleaned_quali_time['q3'] = pd.to_timedelta('00:' + data_cleaned_quali_time['q3'], errors='coerce')

    # convert q1 q2 q3 to milliseconds
    data_cleaned_quali_time['q1'] = data_cleaned_quali_time['q1'].dt.total_seconds() * 1000
    data_cleaned_quali_time['q2'] = data_cleaned_quali_time['q2'].dt.total_seconds() * 1000
    data_cleaned_quali_time['q3'] = data_cleaned_quali_time['q3'].dt.total_seconds() * 1000

    data_cleaned_quali_time['q1'] = data_cleaned_quali_time['q1'].fillna(0)
    data_cleaned_quali_time['q2'] = data_cleaned_quali_time['q2'].fillna(0)
    data_cleaned_quali_time['q3'] = data_cleaned_quali_time['q3'].fillna(0)

    # get fastest time out of q1, q2,q3 for each row and set as qualifying time
    data_cleaned_quali_time['milliseconds_qualification'] = data_cleaned_quali_time[['q1', 'q2', 'q3']].min(axis=1)

    # drop invalid / zero qualification times and duplicates
    data_cleaned_quali_time = data_cleaned_quali_time[data_cleaned_quali_time["milliseconds_qualification"] != 0].copy()
    data_cleaned_quali_time.drop_duplicates(inplace=True)

    # compute median qualification per (circuit, race_year, date)
    median_keys = [k for k in ("circuit", "race_year", "date") if k in data_cleaned_quali_time.columns]
    if median_keys:
        data_median_qualification = (
            data_cleaned_quali_time.groupby(median_keys)["milliseconds_qualification"].median().reset_index()
        )
        data_median_qualification.rename(columns={"milliseconds_qualification": "median_qualification_duration"},
                                         inplace=True)
        data_median = data_cleaned_quali_time.merge(data_median_qualification, on=median_keys, how="left")
        data_median["deviation_from_median"] = data_median["milliseconds_qualification"] - data_median[
            "median_qualification_duration"]
    else:
        data_median = data_cleaned_quali_time.copy()
        data_median["median_qualification_duration"] = np.nan
        data_median["deviation_from_median"] = np.nan

    # round numeric columns
    num_cols = data_median.select_dtypes(include=[np.number]).columns
    data_median[num_cols] = data_median[num_cols].round()

    # compute final_position per race group (if grouping keys exist)
    rank_keys = [k for k in ("race_year", "race_month", "race_day", "circuit") if k in data_median.columns]
    if rank_keys and "deviation_from_median" in data_median.columns:
        data_median["final_position"] = data_median.groupby(rank_keys)["deviation_from_median"].rank(method="min",
                                                                                                     ascending=True)

    # prepare cleaned (drop columns used for metrics, keep parity with other create_* funcs)
    cols_to_drop = [
        "median_qualification_duration",
        "q1",
        "q2",
        "q3",
        "driver_date_of_birth",
        "median_race_duration",
        "race_duration",
        "first_race_date",
        "date",
        "milliseconds_qualification",
        "median_qualification_duration",
        "laps",
        "rain",
        "driverId",
        "constructorId",
        "final_position"
        # keep domain specific columns as needed
    ]
    cleaned = data_median.drop(columns=[c for c in cols_to_drop if c in data_median.columns], errors="ignore").copy()

    cleaned.to_csv(out_dir_path / "cleaned_data_qualifying_with_median.csv", index=False)

    return data_median.reset_index(drop=True), cleaned.reset_index(drop=True)
