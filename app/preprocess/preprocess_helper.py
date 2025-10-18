from pathlib import Path
import pandas as pd

def export_unique_data(
    df: pd.DataFrame,
    name_suffix: str,
    out_dir: str = "data/processed/features_helper",
) -> None:
    """
    Export unique drivers, constructors, circuits tables from mainrace DataFrame.
    """
    project_root = Path(__file__).resolve().parents[2]
    out_dir_path = Path(out_dir) if Path(out_dir).is_absolute() else project_root / out_dir
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Drivers
    drivers_out = df[["driver", "driver_nationality", "driver_date_of_birth", "first_race_date"]].drop_duplicates()
    drivers_out = drivers_out.rename(columns={"driver": "driverRef"})
    drivers_out.to_csv(out_dir_path / f"drivers_{name_suffix}.csv", index=False)

    # Constructors
    constructors_out = df[["constructor", "constructor_nationality"]].drop_duplicates()
    constructors_out = constructors_out.rename(columns={"constructor": "constructorRef"})
    constructors_out.to_csv(out_dir_path / f"constructors_{name_suffix}.csv", index=False)

    # Circuits
    circuits_out = df[["circuit", "circuit_nationality"]].drop_duplicates()
    circuits_out = circuits_out.rename(columns={"circuit": "circuitRef"})
    circuits_out.to_csv(out_dir_path / f"circuits_{name_suffix}.csv", index=False)