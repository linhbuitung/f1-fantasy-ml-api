from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

def serve_qualifying_df(
    raw_dir: Optional[str] = "data/raw",
    processed_dir: Optional[str] = "data/processed",
    year_from: int = 1981,
) -> pd.DataFrame:
    """
    Build the merged DataFrame used as input to
    [`app.preprocess.create_training_datasets`](app/preprocess/preprocess_qualifying.py).

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
    rounds = pd.read_csv(raw_base / "jolpica-dump/formula_one_round.csv")
    round_entries = pd.read_csv(raw_base / "jolpica-dump/formula_one_roundentry.csv")
    sessions = pd.read_csv(raw_base / "jolpica-dump/formula_one_session.csv")
    session_entries = pd.read_csv(raw_base / "jolpica-dump/formula_one_sessionentry.csv")
    team_drivers = pd.read_csv(raw_base / "jolpica-dump/formula_one_teamdriver.csv")
    drivers = pd.read_csv(raw_base / "jolpica-dump/formula_one_driver.csv")
    teams = pd.read_csv(raw_base / "jolpica-dump/formula_one_team.csv")
    circuits = pd.read_csv(raw_base / "jolpica-dump/formula_one_circuit.csv")
    laps = pd.read_csv(raw_base / "jolpica-dump/formula_one_lap.csv")

    circuit_type = pd.read_csv(raw_base / "circuit_type.csv")

    # Notebook merge order (same as Thesis-Qualifying.ipynb)
    df1 = pd.merge(rounds, sessions, how='left', left_on='id', right_on='round_id', suffixes=('_round', '_session'))
    df2 = pd.merge(df1, round_entries, how='left', left_on='id_round', right_on='round_id',
                   suffixes=('', '_round_entry'))
    df2 = df2.rename(columns={'id': 'id_round_entry'})
    df3 = pd.merge(df2, session_entries, how='left', left_on=['id_round_entry', 'id_session'],
                   right_on=['round_entry_id', 'session_id'], suffixes=('', '_session_entry'))
    df3 = df3.rename(columns={'id': 'id_session_entry'})
    df4 = pd.merge(df3, team_drivers, how='left', left_on='team_driver_id', right_on='id',
                   suffixes=('', '_team_driver'))
    df4 = df4.rename(columns={'id': 'id_team_driver'})
    df5 = pd.merge(df4, drivers, how='left', left_on='driver_id', right_on='id', suffixes=('', '_driver'))
    df5 = df5.rename(columns={'id': 'id_driver'})
    df6 = pd.merge(df5, teams, how='left', left_on='team_id', right_on='id', suffixes=('', '_team'))
    df6 = df6.rename(columns={'id': 'id_team'})
    df7 = pd.merge(df6, circuits, how='left', left_on='circuit_id', right_on='id', suffixes=('', '_circuit'))
    df7 = df7.rename(columns={'id': 'id_circuit'})
    df8 = pd.merge(df7, laps, how='left', left_on='id_session_entry', right_on='session_entry_id',
                   suffixes=('', '_lap'))
    df8 = df8.rename(columns={'id': 'id_lap'})

    data = df8

    data = data.drop(
        ['abbreviation', 'altitude', 'average_speed', 'base_team_id', 'car_number', 'circuit_id', 'country', 'detail',
         'date_round', 'fastest_lap_rank', 'forename', 'id_circuit', 'id_round', 'id_round_entry', 'id_session',
         'id_session_entry', 'id_driver', 'id_team', 'id_team_driver', 'id_lap', 'is_deleted', 'is_eligible_for_points',
         'is_entry_fastest_lap', 'latitude', 'locality', 'longitude', 'name', 'name_team', 'name_circuit',
         'nationality', 'nationality_team', 'number', 'number_round', 'number_session', 'permanent_car_number',
         'point_system_id', 'points', 'position', 'position_lap', 'grid', 'race_number', 'role', 'round_entry_id',
         'round_id', 'round_id_round_entry', 'scheduled_laps', 'season_id', 'season_id_team_driver', 'session_entry_id',
         'session_id', 'status', 'surname', 'team_driver_id', 'team_id', 'time', 'wikipedia', 'wikipedia_circuit',
         'wikipedia_driver', 'wikipedia_team', 'time_session_entry', 'laps_completed'], axis=1)

    # rename/normalize columns used in notebook
    rename_map = {
        'date_session':'date',
        'grid':'qualification_position','country_code':'driver_nationality',
        'country_code_team':'constructor_nationality',
        'country_code_circuit':'circuit_nationality',
        'reference_team': 'constructor',
        'reference': 'driver',
        'reference_circuit': 'circuit',
        'date_of_birth':'driver_date_of_birth',
        'time_session_entry': 'race_duration',
        'time_lap': 'lap_duration'
    }
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

    # Take only  column 'type' of value either 'Q1', 'Q2', 'Q3'
    data = data[(data['type'] == 'Q1') | (data['type'] == 'Q2') | (data['type'] == 'Q3')]
    # take only rows where 'is_cancelled_round', 'is_cancelled_session' are 'f'
    data = data[(data['is_cancelled_round'] == 'f')]
    data = data[(data['is_cancelled_session'] == 'f')]

    # drop where is clssified is null or nan
    data = data[data['lap_duration'].notna()]

    # only take is_classified is true
    data.drop(['is_cancelled_round', 'is_cancelled_session', 'type', 'is_classified'], axis=1, inplace=True)

    data['date'] = pd.to_datetime(data['date'])
    data['driver_date_of_birth'] = pd.to_datetime(data['driver_date_of_birth'])

    # get month and day from date into new columns
    data['race_month'] = data['date'].dt.month
    data['race_day'] = data['date'].dt.day
    data['race_year'] = data['date'].dt.year

    data['age_at_gp_in_days'] = abs(data['driver_date_of_birth'] - data['date'])
    data['age_at_gp_in_days'] = data['age_at_gp_in_days'].apply(lambda x: str(x).split(' ')[0]).astype(int)

    first_race_dates = data.groupby('driver_id')['date'].min().reset_index()
    first_race_dates.rename(columns={'date': 'first_race_date'}, inplace=True)
    data = data.merge(first_race_dates, on='driver_id', how='left')
    data = data.drop(['driver_id'], axis=1)

    data['days_since_first_race'] = abs(data['first_race_date'] - data['date'])
    data['days_since_first_race'] = data['days_since_first_race'].apply(lambda x: str(x).split(' ')[0]).astype(int)

    # Merge circuit type
    data = data.merge(circuit_type, how='left', left_on='circuit', right_on='circuit', suffixes=('', '_circuit_type'))

    # data after processing nationalities
    data['driver_home'] = data['driver_nationality'] == data['circuit_nationality']
    data['constructor_home'] = data['constructor_nationality'] == data['circuit_nationality']
    data['driver_home'] = data['driver_home'].apply(lambda x: int(x))
    data['constructor_home'] = data['constructor_home'].apply(lambda x: int(x))

    data = data[data['race_year'] >= year_from]

    data['lap_duration'] = data['lap_duration'].apply(time_to_milliseconds)
    # rename to milliseconds
    data.rename(columns={'lap_duration': 'milliseconds_qualification'}, inplace=True)

    # final housekeeping: drop exact duplicates and return
    data = data.drop_duplicates().reset_index(drop=True)
    return data

def time_to_milliseconds(time_str):
    if pd.isnull(time_str):
        return None
    try:
        # Split milliseconds
        if '.' in time_str:
            time_part, ms_part = time_str.split('.')
            ms = int(ms_part.ljust(3, '0'))  # pad to 3 digits
        else:
            time_part = time_str
            ms = 0
        dt = datetime.strptime(time_part, "%H:%M:%S")
        total_ms = (dt.hour * 3600 + dt.minute * 60 + dt.second) * 1000 + ms
        return total_ms
    except Exception:
        return None

def create_qualifying_training_datasets(
    df: pd.DataFrame,
    out_dir: str = "data/processed",
    deviation_upper: int = 100000,

) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create qualifying training datasets with median qualification duration and deviation from median.
    """
    project_root = Path(__file__).resolve().parents[2]
    out_dir_path = Path(out_dir) if Path(out_dir).is_absolute() else project_root / out_dir
    out_dir_path.mkdir(parents=True, exist_ok=True)

    data_cleaned_quali_time = df.copy()

    columns_to_group = [col for col in data_cleaned_quali_time.columns if col != 'milliseconds_qualification']
    data_cleaned_quali_time = data_cleaned_quali_time.groupby(columns_to_group, as_index=False).agg(
        milliseconds_qualification=('milliseconds_qualification', 'min'),
    )
    # drop if milliseconds_qualification is 0
    data_cleaned_quali_time = data_cleaned_quali_time[data_cleaned_quali_time['milliseconds_qualification'] != 0]

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

    data_median = data_median[data_median['deviation_from_median'] < 1000000]

    # compute final_position per race group (if grouping keys exist)
    rank_keys = [k for k in ("race_year", "race_month", "race_day", "circuit") if k in data_median.columns]
    if rank_keys and "deviation_from_median" in data_median.columns:
        data_median["final_position"] = data_median.groupby(rank_keys)["deviation_from_median"].rank(method="min",
                                                                                                     ascending=True)

    # prepare cleaned (drop columns used for metrics, keep parity with other create_* funcs)
    cols_to_drop = [
        'driver_date_of_birth',
        'date',
        'first_race_date',
        'median_qualification_duration',
        'milliseconds_qualification',
        'final_position'
    ]
    cleaned = data_median.drop(columns=[c for c in cols_to_drop if c in data_median.columns], errors="ignore").copy()

    cleaned.to_csv(out_dir_path / "cleaned_data_qualifying_with_median.csv", index=False)

    return data_median.reset_index(drop=True), cleaned.reset_index(drop=True)

if __name__ == "__main__":
    print("Building qualifying data...")
    df = serve_qualifying_df()  # uses data/raw & data/processed defaults
    data_median, cleaned = create_qualifying_training_datasets(df=df)
    print("Wrote:", Path("data/processed").resolve())