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
    - Saves to drivers.csv with columns: driverRef, driver_nationality
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

    df = drivers
    df.rename(columns={
        'country_code': 'driver_nationality',
        'reference': 'driverRef',
        'date_of_birth': 'driver_date_of_birth',
    }, inplace=True)

    # Specify the date format explicitly
    df['driver_date_of_birth'] = pd.to_datetime(df['driver_date_of_birth'])

    # take only driver_code, driver_date_of_birth and country_code columns
    out = df [["driverRef",'driver_nationality', "driver_date_of_birth"]].drop_duplicates()

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

    df = pd.read_csv(raw_path)

    # take only driver_code and country_code columns
    out = df[["country_code", "reference"]].drop_duplicates()
    out = out.rename(columns={"country_code": "circuit_nationality", "reference": "circuitRef"})

    if save_to:
        save_path = Path(save_to)
        if not save_path.is_absolute():
            save_path = project_root / save_path
        os.makedirs(save_path.parent, exist_ok=True)
        out.to_csv(str(save_path), index=False)
