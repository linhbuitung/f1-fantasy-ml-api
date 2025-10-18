from datetime import datetime
import re
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

from app.preprocess.preprocess_helper import export_unique_data


def serve_mainrace_df(
    raw_dir: Optional[str] = "data/raw",
    processed_dir: Optional[str] = "data/processed",
    year_from: int = 1981,
) -> pd.DataFrame:
    """
    Build the merged DataFrame used as input to
    [`app.preprocess.create_training_datasets`](app/preprocess/preprocess_mainrace.py).
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

    data = df8.copy()

    data = data.drop(
        ['abbreviation', 'altitude', 'average_speed', 'base_team_id', 'car_number', 'circuit_id', 'country',
         'date_session', 'detail', 'fastest_lap_rank', 'forename', 'id_circuit', 'id_round', 'id_round_entry',
         'id_session', 'id_session_entry', 'id_driver', 'id_team', 'id_team_driver', 'id_lap', 'is_deleted',
         'is_eligible_for_points', 'is_entry_fastest_lap', 'latitude', 'locality', 'longitude', 'name', 'name_team',
         'name_circuit', 'nationality', 'nationality_team', 'number', 'number_round', 'number_session',
         'permanent_car_number', 'point_system_id', 'points', 'position', 'position_lap', 'race_number', 'role',
         'round_entry_id', 'round_id', 'round_id_round_entry', 'scheduled_laps', 'season_id', 'season_id_team_driver',
         'session_entry_id', 'session_id', 'status', 'surname', 'team_driver_id', 'team_id', 'time', 'wikipedia',
         'wikipedia_circuit', 'wikipedia_driver', 'wikipedia_team'], axis=1)

    # rename/normalize columns used in notebook
    rename_map = {
        'date_round':'date',
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

    # Take only  column 'type' of value 'R' as race
    data = data[data['type'] == 'R']
    # take only rows where 'is_cancelled_round', 'is_cancelled_session' are 'f'
    data = data[(data['is_cancelled_round'] == 'f')]
    data = data[(data['is_cancelled_session'] == 'f')]

    data.drop(['is_cancelled_round', 'is_cancelled_session', 'type'], axis=1, inplace=True)

    # drop where is clssified is null or nan
    data = data[data['is_classified'].notna()]

    # Specify the date format explicitly
    data['date'] = pd.to_datetime(data['date'])
    data['driver_date_of_birth'] = pd.to_datetime(data['driver_date_of_birth'])

    # get month and day from date into new columns
    data['race_month'] = data['date'].dt.month
    data['race_day'] = data['date'].dt.day
    data['race_year'] = data['date'].dt.year

    # compute age and first race fields (notebook logic)
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

    # take only date from year
    data = data[data['race_year'] >= year_from]

    # data after processing nationalities
    data['driver_home'] = data['driver_nationality'] == data['circuit_nationality']
    data['constructor_home'] = data['constructor_nationality'] == data['circuit_nationality']
    data['driver_home'] = data['driver_home'].apply(lambda x: int(x))
    data['constructor_home'] = data['constructor_home'].apply(lambda x: int(x))

    data['milliseconds'] = data['race_duration'].apply(time_to_milliseconds)
    # fill milliseconds null with 0
    data['milliseconds'] = data['milliseconds'].fillna(0)
    data['milliseconds_laptime'] = data['lap_duration'].apply(time_to_milliseconds)
    data['milliseconds_laptime'] = data['milliseconds_laptime'].fillna(0)

    # drop race_duration and lap_duration
    data = data.drop(['race_duration', 'lap_duration'], axis=1)

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


def create_mainrace_training_datasets(
    df: pd.DataFrame,
    out_dir: str = "data/processed",
    min_laps_threshold: int = 10,
    deviation_lower: int = -110000,
    deviation_upper: int = 612000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Implements the preprocessing steps from research/Thesis.ipynb
    """
    project_root = Path(__file__).resolve().parents[2]
    out_dir_path = Path(out_dir) if Path(out_dir).is_absolute() else project_root / out_dir
    out_dir_path.mkdir(parents=True, exist_ok=True)

    data = df.copy()
    
    # build feature data from helper
    export_unique_data(data, name_suffix="mainrace")

    # 1) Group by all columns except 'milliseconds_laptime' and aggregate
    cols_to_group = [c for c in data.columns if c != "milliseconds_laptime"]
    grouped = (
        data.groupby(cols_to_group, as_index=False)
        .agg(milliseconds_laptime=("milliseconds_laptime", "sum"),
             laps_count=("milliseconds_laptime", "count"))
    )

    # 2) Numeric conversions and fill
    grouped['milliseconds_laptime'] = pd.to_numeric(grouped['milliseconds_laptime'],
                                                                     errors='coerce')
    grouped['milliseconds'] = pd.to_numeric(grouped['milliseconds'], errors='coerce')
    # Replace '\N' with 0 in both columns
    grouped['milliseconds_laptime'] = grouped['milliseconds_laptime'].fillna(0)
    grouped['milliseconds'] = grouped['milliseconds'].fillna(0)

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

    # 5) laps adjustment: keep existing 'laps_completed' but ensure >= laps_count
    grouped['laps_completed'] = grouped.apply(
        lambda row: row['laps_completed'] if row['laps_completed'] >= row['laps_count'] else row['laps_count'], axis=1
    )

    grouped.drop(columns=["laps_count"], inplace=True, errors="ignore")

    # 6) compute max_laps per (circuit, race_year, date)
    grouped["laps_completed"] = pd.to_numeric(grouped.get("laps_completed"), errors="coerce")
    median_keys = [k for k in ("circuit", "race_year", "date") if k in grouped.columns]
    if median_keys:
        grouped["max_laps"] = grouped.groupby(median_keys)["laps_completed"].transform("max")

    # 7) prepare data_median working frame
    data_median = grouped.copy()
    # drop duplicates (not harmful)
    data_median = data_median.drop_duplicates().reset_index(drop=True)
    # drop where laps < min_laps_threshold
    data_median = data_median[data_median["laps_completed"] >= min_laps_threshold].copy()

    # 8) compute additional_laps and final_race_duration for finished or +N Lap statuses
    data_median["additional_laps"] = (data_median["max_laps"] - data_median["laps_completed"]).abs().fillna(0)

    data_median['laps'] = data_median['max_laps']
    # Apply the condition
    condition = data_median['is_classified'].eq('t')

    data_median.loc[condition, 'final_race_duration'] = (
            data_median['race_duration'] +
            data_median['race_duration'] / data_median['laps_completed'] * data_median['additional_laps']
    )
    data_median['final_race_duration'] = data_median['final_race_duration'].fillna(data_median['race_duration'])

    # 9) median per circuit/year/date and deviation
    if median_keys:
        data_median_race_duration = data_median.groupby(median_keys)[
            'final_race_duration'].median().reset_index()
        data_median_race_duration.rename(columns={'final_race_duration': 'median_race_duration'}, inplace=True)

        data_median = data_median.merge(data_median_race_duration, on=median_keys, how="left")
        data_median['deviation_from_median'] = data_median['final_race_duration'] - data_median['median_race_duration']
    else:
        data_median["median_race_duration"] = np.nan
        data_median["deviation_from_median"] = np.nan

    # 10) filter statuses like notebook and drop small races
    data_median = data_median[data_median['is_classified'].eq('t')]

    # 11) round numeric columns
    num_cols = data_median.select_dtypes(include=[np.number]).columns
    data_median[num_cols] = data_median[num_cols].round()


    # 12) filter extreme deviations
    data_median = data_median[
        (data_median['deviation_from_median'] > deviation_lower) & (data_median['deviation_from_median'] < deviation_upper)]

    # 13) final_position ranking per race for export (used later for evaluation)
    rank_keys = [k for k in ("race_year", "race_month", "race_day", "circuit") if k in data_median.columns]
    if rank_keys and "deviation_from_median" in data_median.columns:
        data_median["final_position"] = data_median.groupby(rank_keys)["deviation_from_median"].rank(method="min", ascending=True)

    # 14) create cleaned_data_median by dropping columns used only for computing metrics (mirror notebook)
    cols_to_drop = [
        'driver_date_of_birth',
        'date',
        'first_race_date',
        'additional_laps',
        'is_classified',
        'laps_completed',
        'milliseconds',
       'milliseconds_laptime',
        'race_duration',
        'max_laps',
        'final_race_duration',
        'median_race_duration'
    ]
    cleaned = data_median.drop(columns=[c for c in cols_to_drop if c in data_median.columns], errors="ignore").copy()

    # drop driverId/constructorId if present (not used in models)

    cleaned.to_csv(out_dir_path / "cleaned_data_main_race_with_median.csv", index=False)

    return data_median.reset_index(drop=True), cleaned.reset_index(drop=True)

if __name__ == "__main__":
    print("Building mainrace data...")
    df = serve_mainrace_df()  # uses data/raw & data/processed defaults
    data_median, cleaned = create_mainrace_training_datasets(df=df)
    print("Wrote:", Path("data/processed").resolve())