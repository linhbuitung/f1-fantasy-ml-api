import os
from pathlib import Path
from typing import  List, Optional
import pandas as pd
from app.schemas.dto import Race

def build_all_general_processed_data():
    build_driver_country_table()
    build_constructor_country_table()
    build_circuit_country_table()

def build_driver_country_table(
    drivers_path: Optional[str] = None,
    save_to: str = "data/processed/drivers.csv",
):
    """
    Build and persist unique driver table with alpha-3 nationality.
    - If drivers_path is None, loads data/raw/jolpica-dump/formula_one_driver.csv.
    - Assumes the driver CSV has columns: reference (driverRef), country_code
    - Saves to drivers.csv with columns: driverRef, driver_nationality, driver_date_of_birth, first_race_date
    """
    project_root = Path(__file__).resolve().parents[2]
    # Resolve drivers CSV
    if drivers_path is None:
        raw_path = project_root / "data" / "raw" / "jolpica-dump" / "formula_one_driver.csv"
    else:
        raw_path = Path(drivers_path)
        if not raw_path.is_absolute():
            raw_path = project_root / raw_path

    drivers = pd.read_csv(raw_path)
    rounds = pd.read_csv(project_root / "data" / "raw" /"jolpica-dump" / "formula_one_round.csv")
    round_entries = pd.read_csv(project_root / "data" / "raw" /"jolpica-dump" / "formula_one_roundentry.csv")
    sessions = pd.read_csv(project_root / "data" / "raw" /"jolpica-dump" / "formula_one_session.csv")
    session_entries = pd.read_csv(project_root / "data" / "raw" /"jolpica-dump" / "formula_one_sessionentry.csv")
    team_drivers = pd.read_csv(project_root / "data" / "raw" /"jolpica-dump" / "formula_one_teamdriver.csv")

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

    data = df5

    # rename/normalize columns used in notebook
    rename_map = {
        'date_round': 'date',
        'country_code': 'driver_nationality',
        'country_code_team': 'constructor_nationality',
        'country_code_circuit': 'circuit_nationality',
        'reference_team': 'constructor',
        'reference': 'driverRef',
        'date_of_birth': 'driver_date_of_birth',
    }
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

    # Specify the date format explicitly
    data['date'] = pd.to_datetime(data['date'])
    data['driver_date_of_birth'] = pd.to_datetime(data['driver_date_of_birth'])

    first_race_dates = data.groupby('driver_id')['date'].min().reset_index()
    first_race_dates.rename(columns={'date': 'first_race_date'}, inplace=True)
    data = data.merge(first_race_dates, on='driver_id', how='left')

    # take only driver_code, driver_date_of_birth and country_code columns
    out = data [["driverRef",'driver_nationality', "driver_date_of_birth", 'first_race_date']].drop_duplicates()
    # drop rows with null values
    out = out.dropna()
    if save_to:
        save_path = Path(save_to)
        if not save_path.is_absolute():
            save_path = project_root / save_path
        os.makedirs(save_path.parent, exist_ok=True)
        out.to_csv(str(save_path), index=False)

def build_constructor_country_table(
    constructors_path: Optional[str] = None,
    save_to: str = "data/processed/constructors.csv",
):
    """
    Build and persist unique constructor table with alpha-3 nationality.
    - If constructors_path is None, loads data/raw/jolpica-dump/formula_one_team.csv.
    - Assumes the constructor CSV has columns: reference (constructorRef), country_code
    - Saves to constructors.csv with columns: constructorRef, constructor_nationality
    """
    project_root = Path(__file__).resolve().parents[2]
    # Resolve drivers CSV
    if constructors_path is None:
        raw_path = project_root / "data" / "raw" / "jolpica-dump" / "formula_one_team.csv"
    else:
        raw_path = Path(constructors_path)
        if not raw_path.is_absolute():
            raw_path = project_root / raw_path

    df = pd.read_csv(raw_path)

    # take only driver_code and country_code columns
    out = df[["country_code", "reference"]].drop_duplicates()
    out = out.rename(columns={"country_code": "constructor_nationality", "reference": "constructorRef"})
    # drop rows with null values
    out = out.dropna()
    if save_to:
        save_path = Path(save_to)
        if not save_path.is_absolute():
            save_path = project_root / save_path
        os.makedirs(save_path.parent, exist_ok=True)
        out.to_csv(str(save_path), index=False)

def build_circuit_country_table(
    circuits_path: Optional[str] = None,
    save_to: str = "data/processed/circuits.csv",
):
    """
    Build and persist unique circuit table with alpha-3 nationality.

    - If circuits_path is None, loads data/raw/circuits.csv.
    """
    project_root = Path(__file__).resolve().parents[2]
    # Resolve drivers CSV
    if circuits_path is None:
        raw_path = project_root / "data" / "raw" / "jolpica-dump" / "formula_one_circuit.csv"
    else:
        raw_path = Path(circuits_path)
        if not raw_path.is_absolute():
            raw_path = project_root / raw_path

    circuit_type = pd.read_csv(project_root / "data" / "raw" / "circuit_type.csv")

    df = pd.read_csv(raw_path)
    # merge with circuit_type by reference
    df = pd.merge(df, circuit_type, how='left', left_on='reference', right_on='circuit', suffixes=('', '_circuit_type'))

    # take only driver_code and country_code columns
    out = df[["country_code", "reference", "type_circuit"]].drop_duplicates()
    out = out.rename(columns={"country_code": "circuit_nationality", "reference": "circuitRef"})
    # drop rows with null values
    out = out.dropna()
    if save_to:
        save_path = Path(save_to)
        if not save_path.is_absolute():
            save_path = project_root / save_path
        os.makedirs(save_path.parent, exist_ok=True)
        out.to_csv(str(save_path), index=False)
