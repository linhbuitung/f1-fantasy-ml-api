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

    race_weather = pd.read_csv(raw_base / "race_weather.csv")
    circuit_type = pd.read_csv(raw_base / "circuit_type.csv")

    # Notebook merge order (same as Thesis.ipynb)
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

    data = data.drop(['abbreviation', 'altitude', 'average_speed', 'base_team_id',
                      'car_number', 'circuit_id', 'country', 'date_session', 'detail',
                      'fastest_lap_rank', 'forename', 'id_circuit', 'id_round',
                      'id_round_entry', 'id_session', 'id_session_entry', 'id_driver',
                      'id_team', 'id_team_driver', 'id_lap', 'is_deleted',
                      'is_eligible_for_points', 'is_entry_fastest_lap', 'laps_completed',
                      'latitude', 'locality', 'longitude', 'name', 'name_team',
                      'name_circuit', 'nationality', 'nationality_team', 'number',
                      'number_round', 'number_session', 'permanent_car_number',
                      'point_system_id', 'points' , 'position', 'position_lap',
                      'race_number', 'role', 'round_entry_id', 'round_id',
                      'round_id_round_entry', 'scheduled_laps', 'season_id',
                      'season_id_team_driver', 'session_entry_id', 'session_id', 'status',
                      'surname', 'team_driver_id', 'team_id', 'time', 'wikipedia',
                      'wikipedia_circuit', 'wikipedia_driver', 'wikipedia_team',
                      'time_lap' ,'time_session_entry' ],axis=1)

    # rename/normalize columns used in notebook
    rename_map = {
        'date_round':'date',
        'grid':'qualification_position',
        'country_code':'driver_nationality',
        'country_code_team':'constructor_nationality'
        ,'country_code_circuit':'circuit_nationality',
        'reference_team': 'constructor',
        'reference': 'driver',
        'reference_circuit': 'circuit',
        'date_of_birth':'driver_date_of_birth'
    }
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

    # Take only  column 'type' of value 'R' as race
    data = data[data['type'] == 'R']
    # take only rows where 'is_cancelled_round', 'is_cancelled_session' are 'f'
    data = data[(data['is_cancelled_round'] == 'f')]
    data = data[(data['is_cancelled_session'] == 'f')]

    data.drop(['is_cancelled_round', 'is_cancelled_session', 'type'], axis=1, inplace=True)

    # drop where is classified is null or nan
    data = data[data['is_classified'].notna()]

    # Specify the date format explicitly
    data['date'] = pd.to_datetime(data['date'])
    data['driver_date_of_birth'] = pd.to_datetime(data['driver_date_of_birth'])

    # get month and day from date into new columns
    data['race_month'] = data['date'].dt.month
    data['race_day'] = data['date'].dt.day
    data['race_year'] = data['date'].dt.year

    # driver DOB -> datetime
    data['age_at_gp_in_days'] = abs(data['driver_date_of_birth'] - data['date'])
    data['age_at_gp_in_days'] = data['age_at_gp_in_days'].apply(lambda x: str(x).split(' ')[0]).astype(int)

    first_race_dates = data.groupby('driver_id')['date'].min().reset_index()
    first_race_dates.rename(columns={'date': 'first_race_date'}, inplace=True)
    data = data.merge(first_race_dates, on='driver_id', how='left')
    data = data.drop(['driver_id'], axis=1)

    data['days_since_first_race'] = abs(data['first_race_date'] - data['date'])
    data['days_since_first_race'] = data['days_since_first_race'].apply(lambda x: str(x).split(' ')[0]).astype(int)

    # load race_weather csv, join by race date
    race_weather['date'] = pd.to_datetime(race_weather['date'])
    data = data.merge(race_weather, how='left', left_on='date', right_on='date', suffixes=('', '_race_weather'))
    data = data[data['weather'].notna()]

    # create a rain column where if the weather is 'Rain or 'Changeable' or 'Very changeable' then 1 else 0
    data['rain'] = data['weather'].apply(lambda x: 1 if x in ['Rain', 'Changeable', 'Very changeable'] else 0)
    # drop the weather column
    data = data.drop(['weather'], axis=1)

    # Merge circuit type
    data = data.merge(circuit_type, how='left', left_on='circuit', right_on='circuit', suffixes=('', '_circuit_type'))

    data = data[data['race_year'] >= year_from]

    # data after processing nationalities
    data['driver_home'] = data['driver_nationality'] == data['circuit_nationality']
    data['constructor_home'] = data['constructor_nationality'] == data['circuit_nationality']
    data['driver_home'] = data['driver_home'].apply(lambda x: int(x))
    data['constructor_home'] = data['constructor_home'].apply(lambda x: int(x))

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

    # dnf is 1 if is_classified is = 'f' else 0
    data_cleaned_status['dnf'] = data_cleaned_status['is_classified'].apply(lambda x: 0 if x == 't' else 1)
    cleaned = data_cleaned_status.drop(['is_classified'], axis=1)

    columns_to_drop = ['driver_date_of_birth', 'date', 'first_race_date']
    cleaned = cleaned.drop(columns=[c for c in columns_to_drop if c in cleaned.columns], errors="ignore")

    cleaned.to_csv(out_dir_path / "cleaned_data_status.csv", index=False)

    return data_cleaned_status.reset_index(drop=True), cleaned.reset_index(drop=True)

if __name__ == "__main__":
    print("Building status data...")
    df = serve_status_df()  # uses data/raw & data/processed defaults
    data_median, cleaned = create_status_training_datasets(df=df)
    print("Wrote:", Path("data/processed").resolve())