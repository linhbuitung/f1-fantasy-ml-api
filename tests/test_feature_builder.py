from pathlib import Path
import sys
import json
import pandas as pd
from src.services.feature_builder import build_main_race_features_from_dto

def _write_csv(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)

def test_build_features_from_dto_minimal(tmp_path, monkeypatch):
    # arrange: create expected data files under tmp_path (repo root)
    project_root = tmp_path
    data_proc = project_root / "data" / "processed"
    data_raw = project_root / "data" / "raw"
    data_proc.mkdir(parents=True, exist_ok=True)
    data_raw.mkdir(parents=True, exist_ok=True)

    # drivers.csv with driverRef, nationality and dob
    drivers_csv = data_proc / "drivers.csv"
    drivers_content = "driverId,driverRef,forename,surname,driver_date_of_birth,first_race_date,nationality\n"
    drivers_content += "1,hamilton,Lewis,Hamilton,1985-01-07,2007-03-18,GBR\n"
    _write_csv(drivers_csv, drivers_content)

    # constructors.csv
    constructors_csv = data_proc / "constructors.csv"
    constructors_content = "constructor,constructorRef,constructor_nationality\n"
    constructors_content += "Ferrari,ferrari,ITA\n"
    _write_csv(constructors_csv, constructors_content)

    # circuits.csv
    circuits_csv = data_proc / "circuits.csv"
    circuits_content = "circuit,circuitRef,circuit_nationality,type_circuit\n"
    circuits_content += "silverstone,silverstone,GBR,Race circuit\n"
    _write_csv(circuits_csv, circuits_content)

    # results.csv (empty header is fine)
    results_csv = data_raw / "results.csv"
    _write_csv(results_csv, "raceId,driverId\n")

    # switch CWD so feature_builder reads tmp_path/data/...
    monkeypatch.chdir(project_root)

    # minimal DTO (note driver matches driverRef)
    dto = {
        "qualification_position": 10,
        "laps": 58,
        "constructor": "ferrari",
        "circuit": "silverstone",
        "driver": "hamilton",
        "race_date": "2023-07-14",
        "rain": 0,
    }

    # act
    features = build_main_race_features_from_dto(dto)

    # assert presence of core keys
    expected_keys = {
        "qualification_position",
        "laps",
        "constructor",
        "circuit",
        "driver",
        "circuit_nationality",
        "driver_nationality",
        "constructor_nationality",
        "race_year",
        "race_month",
        "race_day",
        "driver_home",
        "constructor_home",
    }
    assert expected_keys.issubset(set(features.keys()))

    # driver_home should be true because driver_nationality GBR == circuit_nationality GBR
    assert features["driver_nationality"] == "GBR"
    assert features["circuit_nationality"] == "GBR"
    assert features["driver_home"] == 1

    # age_at_gp_in_days computed from dob and race_date (approx)
    assert "age_at_gp_in_days" in features
    dob = pd.to_datetime("1985-01-07")
    race_date = pd.to_datetime("2023-07-14")
    expected_age = int((race_date - dob).days)
    assert int(features["age_at_gp_in_days"]) == expected_age