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

    # get drivers from csv
    processed_drivers_path = project_root / "data/processed/drivers.csv"
    processed_drivers_df = pd.read_csv(processed_drivers_path)

    drivers_out = df[["driver"]].drop_duplicates()
    # merge with processed drivers
    drivers_out = drivers_out.merge(processed_drivers_df, how="left", left_on="driver", right_on="driverRef", suffixes=('', '_proc'))
    # drop driver column from processed
    drivers_out = drivers_out.drop(columns=["driver"])

    drivers_out.to_csv(out_dir_path / f"drivers_{name_suffix}.csv", index=False)

    # Constructors
    constructors_out = df[["constructor", "constructor_nationality"]].drop_duplicates()
    constructors_out = constructors_out.rename(columns={"constructor": "constructorRef"})
    constructors_out.to_csv(out_dir_path / f"constructors_{name_suffix}.csv", index=False)

    # Circuits
    circuits_out = df[["circuit", "circuit_nationality"]].drop_duplicates()
    circuits_out = circuits_out.rename(columns={"circuit": "circuitRef"})
    circuits_out.to_csv(out_dir_path / f"circuits_{name_suffix}.csv", index=False)