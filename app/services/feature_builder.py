import os
from pathlib import Path
from typing import Dict
import pandas as pd
from datetime import datetime

project_root = Path(__file__).resolve().parents[2]  # repo root (.. / .. from this file)
DATA_DIR = Path(os.environ.get("MODEL_DIR", str(project_root / "data")))

def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str)

def validate_features_pickable(driver: str, constructor: str, circuit: str, type: str) -> bool:
    drivers = _load_csv(DATA_DIR / "processed" / "features_helper" / f"drivers_{type}.csv")
    constructors = _load_csv(DATA_DIR / "processed" / "features_helper" / f"constructors_{type}.csv")
    circuits = _load_csv(DATA_DIR / "processed" / "features_helper" / f"circuits_{type}.csv")

    driver_valid = not drivers.empty and (drivers.get("driverRef", "").fillna("").str.lower() == driver.lower()).any()
    constructor_valid = not constructors.empty and (constructors.get("constructorRef", "").fillna("").str.lower() == constructor.lower()).any()
    circuit_valid = not circuits.empty and (circuits.get("circuitRef", "").fillna("").str.lower() == circuit.lower()).any()

    return driver_valid and constructor_valid and circuit_valid


def build_main_race_features_from_dto(dto: Dict) -> Dict:
    """
    Input: dict with keys matching PredictInput (driver, constructor, circuit, race_date (date), ...)
    Output: dict with full feature set expected by the pipeline (age_at_gp_in_days, driver_nationality, etc.)
    """
    # lookup tables (small)
    drivers = _load_csv(DATA_DIR / "processed" / "drivers.csv")
    constructors = _load_csv(DATA_DIR / "processed" / "constructors.csv")
    circuits = _load_csv(DATA_DIR / "processed" / "circuits.csv")

    # canonical inputs
    driver_ref = str(dto.get("driver")).strip()
    constructor_ref = str(dto.get("constructor")).strip()
    circuit_ref = str(dto.get("circuit")).strip()
    race_date = pd.to_datetime(dto.get("race_date"))

    # validate pickable
    if not validate_features_pickable(driver_ref, constructor_ref, circuit_ref, "mainrace"):
        raise ValueError("One or more of driver, constructor, or circuit is not pickable data.")

    drv_row = pd.DataFrame()
    if not drivers.empty:
        drv_row = drivers[drivers.get("driverRef", "").fillna("").str.lower() == driver_ref.lower()]
        if drv_row.empty:
            # try composite name fallback
            full = drivers.get("forename", "").fillna("") + " " + drivers.get("surname", "").fillna("")
            drv_row = drivers[full.str.lower() == driver_ref.lower()]

    driver_nationality = drv_row.iloc[0]["driver_nationality"] if not drv_row.empty and "driver_nationality" in drv_row.columns else None

    driver_date_of_birth = None
    if not drv_row.empty and "driver_date_of_birth" in drv_row.columns:
        try:
            driver_date_of_birth = pd.to_datetime(drv_row.iloc[0]["driver_date_of_birth"], errors="coerce")
        except Exception:
            driver_date_of_birth = None

    age_at_gp_in_days = (race_date - driver_date_of_birth).days if driver_date_of_birth is not None else None

    first_race_date = None
    if not drv_row.empty and "first_race_date" in drv_row.columns:
        try:
            first_race_date = pd.to_datetime(drv_row.iloc[0]["first_race_date"], errors="coerce")
        except Exception:
            first_race_date = None

    days_since_first_race = (race_date - first_race_date).days if first_race_date is not None else None
    # constructor nationality
    cons_row = pd.DataFrame()
    if not constructors.empty:
        mask = constructors.get("constructorRef", "").fillna("").str.lower() == constructor_ref.lower()
        cons_row = constructors[mask]
    constructor_nationality = cons_row.iloc[0].get("constructor_nationality") if not cons_row.empty else None

    # circuit nationality / type: prefer circuits.name or circuitRef
    circ_row = pd.DataFrame()
    if not circuits.empty:
        mask = circuits.get("circuitRef", "").fillna("").str.lower() == circuit_ref.lower()
        circ_row = circuits[mask]
    circuit_country = circ_row.iloc[0].get("circuit_nationality") if not circ_row.empty and "circuit_nationality" in circ_row.columns else None
    type_circuit = circ_row.iloc[0].get("type_circuit") if not circ_row.empty and "type_circuit" in circ_row.columns else None

    # build final features dict (fill None -> pd.NA or defaults)
    features = {
        "qualification_position": int(dto["qualification_position"]),
        "laps": int(dto["laps"]),
        "constructor": constructor_ref,
        "circuit": circuit_ref,
        "type_circuit": type_circuit or dto.get("type_circuit"),
        "driver": driver_ref,
        "circuit_nationality": circuit_country,
        "driver_nationality": driver_nationality,
        "constructor_nationality": constructor_nationality,
        "race_year": int(race_date.year),
        "race_month": int(race_date.month),
        "race_day": int(race_date.day),
        "rain": int(dto.get("rain", 0)),
        "driver_home": 1 if driver_nationality and circuit_country and driver_nationality == circuit_country else 0,
        "constructor_home": 1 if constructor_nationality and circuit_country and constructor_nationality == circuit_country else 0,
    }
    if age_at_gp_in_days is not None:
        features["age_at_gp_in_days"] = int(age_at_gp_in_days)
    if days_since_first_race is not None:
        features["days_since_first_race"] = int(days_since_first_race)

    return features

def build_qualifying_features_from_dto(dto: Dict) -> Dict:
    """
    Input: dict with keys matching PredictInput (driver, constructor, circuit, race_date (date), ...)
    Output: dict with full feature set expected by the pipeline (age_at_gp_in_days, driver_nationality, etc.)
    """
    # lookup tables (small)
    drivers = _load_csv(DATA_DIR / "processed" / "drivers.csv")
    constructors = _load_csv(DATA_DIR / "processed" / "constructors.csv")
    circuits = _load_csv(DATA_DIR / "processed" / "circuits.csv")

    # canonical inputs
    driver_ref = str(dto.get("driver")).strip()
    constructor_ref = str(dto.get("constructor")).strip()
    circuit_ref = str(dto.get("circuit")).strip()
    race_date = pd.to_datetime(dto.get("race_date"))

    # validate pickable
    if not validate_features_pickable(driver_ref, constructor_ref, circuit_ref, "qualifying"):
        raise ValueError("One or more of driver, constructor, or circuit is not pickable data.")

    # look up driver row (try driverRef then forename+surname)
    drv_row = pd.DataFrame()
    if not drivers.empty:
        drv_row = drivers[drivers.get("driverRef", "").fillna("").str.lower() == driver_ref.lower()]
        if drv_row.empty:
            # try composite name fallback
            full = drivers.get("forename", "").fillna("") + " " + drivers.get("surname", "").fillna("")
            drv_row = drivers[full.str.lower() == driver_ref.lower()]

    driver_nationality = drv_row.iloc[0]["driver_nationality"] if not drv_row.empty and "driver_nationality" in drv_row.columns else None

    driver_date_of_birth = None
    if not drv_row.empty and "driver_date_of_birth" in drv_row.columns:
        try:
            driver_date_of_birth = pd.to_datetime(drv_row.iloc[0]["driver_date_of_birth"], errors="coerce")
        except Exception:
            driver_date_of_birth = None

    age_at_gp_in_days = (race_date - driver_date_of_birth).days if driver_date_of_birth is not None else None

    first_race_date = None
    if not drv_row.empty and "first_race_date" in drv_row.columns:
        try:
            first_race_date = pd.to_datetime(drv_row.iloc[0]["first_race_date"], errors="coerce")
        except Exception:
            first_race_date = None

    days_since_first_race = (race_date - first_race_date).days if first_race_date is not None else None

    # constructor nationality
    cons_row = pd.DataFrame()
    if not constructors.empty:
        mask = (
                       constructors.get("constructor", "").fillna("").str.lower() == constructor_ref.lower()
               ) | (
                       constructors.get("constructorRef", "").fillna("").str.lower() == constructor_ref.lower()
               )
        cons_row = constructors[mask]
    constructor_nationality = cons_row.iloc[0].get("constructor_nationality") if not cons_row.empty else None

    # circuit nationality / type: prefer circuits.name or circuitRef
    circ_row = pd.DataFrame()
    if not circuits.empty:
        mask = (
                       circuits.get("circuit", "").fillna("").str.lower() == circuit_ref.lower()
               ) | (
                       circuits.get("circuitRef", "").fillna("").str.lower() == circuit_ref.lower()
               )
        circ_row = circuits[mask]
    circuit_country = circ_row.iloc[0].get("circuit_nationality") if not circ_row.empty and "circuit_nationality" in circ_row.columns else None
    type_circuit = circ_row.iloc[0].get("type_circuit") if not circ_row.empty and "type_circuit" in circ_row.columns else None

    # build final features dict (fill None -> pd.NA or defaults)
    features = {
        "constructor": constructor_ref,
        "circuit": circuit_ref,
        "type_circuit": type_circuit or dto.get("type_circuit"),
        "driver": driver_ref,
        "circuit_nationality": circuit_country,
        "driver_nationality": driver_nationality,
        "constructor_nationality": constructor_nationality,
        "race_year": int(race_date.year),
        "race_month": int(race_date.month),
        "race_day": int(race_date.day),
        "driver_home": 1 if driver_nationality and circuit_country and driver_nationality == circuit_country else 0,
        "constructor_home": 1 if constructor_nationality and circuit_country and constructor_nationality == circuit_country else 0,
    }
    if age_at_gp_in_days is not None:
        features["age_at_gp_in_days"] = int(age_at_gp_in_days)
    if days_since_first_race is not None:
        features["days_since_first_race"] = int(days_since_first_race)

    return features

def build_status_features_from_dto(dto: Dict) -> Dict:
    """
    Input: dict with keys matching PredictInput (driver, constructor, circuit, race_date (date), ...)
    Output: dict with full feature set expected by the pipeline (age_at_gp_in_days, driver_nationality, etc.)
    """
    # lookup tables (small)
    drivers = _load_csv(DATA_DIR / "processed" / "drivers.csv")
    constructors = _load_csv(DATA_DIR / "processed" / "constructors.csv")
    circuits = _load_csv(DATA_DIR / "processed" / "circuits.csv")

    # canonical inputs
    driver_ref = str(dto.get("driver")).strip()
    constructor_ref = str(dto.get("constructor")).strip()
    circuit_ref = str(dto.get("circuit")).strip()
    race_date = pd.to_datetime(dto.get("race_date"))

    # validate pickable
    if not validate_features_pickable(driver_ref, constructor_ref, circuit_ref, "status"):
        raise ValueError("One or more of driver, constructor, or circuit is not pickable data.")

    # look up driver row (try driverRef then forename+surname)
    drv_row = pd.DataFrame()
    if not drivers.empty:
        drv_row = drivers[drivers.get("driverRef", "").fillna("").str.lower() == driver_ref.lower()]
        if drv_row.empty:
            # try composite name fallback
            full = drivers.get("forename", "").fillna("") + " " + drivers.get("surname", "").fillna("")
            drv_row = drivers[full.str.lower() == driver_ref.lower()]

    driver_nationality = drv_row.iloc[0]["driver_nationality"] if not drv_row.empty and "driver_nationality" in drv_row.columns else None

    driver_date_of_birth = None
    if not drv_row.empty and "driver_date_of_birth" in drv_row.columns:
        try:
            driver_date_of_birth = pd.to_datetime(drv_row.iloc[0]["driver_date_of_birth"], errors="coerce")
        except Exception:
            driver_date_of_birth = None

    age_at_gp_in_days = (race_date - driver_date_of_birth).days if driver_date_of_birth is not None else None

    first_race_date = None
    if not drv_row.empty and "first_race_date" in drv_row.columns:
        try:
            first_race_date = pd.to_datetime(drv_row.iloc[0]["first_race_date"], errors="coerce")
        except Exception:
            first_race_date = None

    days_since_first_race = (race_date - first_race_date).days if first_race_date is not None else None

    # constructor nationality
    cons_row = pd.DataFrame()
    if not constructors.empty:
        mask = (
                       constructors.get("constructor", "").fillna("").str.lower() == constructor_ref.lower()
               ) | (
                       constructors.get("constructorRef", "").fillna("").str.lower() == constructor_ref.lower()
               )
        cons_row = constructors[mask]
    constructor_nationality = cons_row.iloc[0].get("constructor_nationality") if not cons_row.empty else None

    # circuit nationality / type: prefer circuits.name or circuitRef
    circ_row = pd.DataFrame()
    if not circuits.empty:
        mask = (
                       circuits.get("circuit", "").fillna("").str.lower() == circuit_ref.lower()
               ) | (
                       circuits.get("circuitRef", "").fillna("").str.lower() == circuit_ref.lower()
               )
        circ_row = circuits[mask]
    circuit_country = circ_row.iloc[0].get("circuit_nationality") if not circ_row.empty and "circuit_nationality" in circ_row.columns else None
    type_circuit = circ_row.iloc[0].get("type_circuit") if not circ_row.empty and "type_circuit" in circ_row.columns else None

    # build final features dict (fill None -> pd.NA or defaults)
    features = {
        "qualification_position": int(dto["qualification_position"]),
        "constructor": constructor_ref,
        "circuit": circuit_ref,
        "type_circuit": type_circuit or dto.get("type_circuit"),
        "driver": driver_ref,
        "circuit_nationality": circuit_country,
        "driver_nationality": driver_nationality,
        "constructor_nationality": constructor_nationality,
        "race_year": int(race_date.year),
        "race_month": int(race_date.month),
        "race_day": int(race_date.day),
        "rain": int(dto.get("rain", 0)),
        "driver_home": 1 if driver_nationality and circuit_country and driver_nationality == circuit_country else 0,
        "constructor_home": 1 if constructor_nationality and circuit_country and constructor_nationality == circuit_country else 0,
    }
    if age_at_gp_in_days is not None:
        features["age_at_gp_in_days"] = int(age_at_gp_in_days)
    if days_since_first_race is not None:
        features["days_since_first_race"] = int(days_since_first_race)

    return features