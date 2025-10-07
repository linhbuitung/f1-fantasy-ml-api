import os
from pathlib import Path
from typing import  List, Optional
import pandas as pd
from app.schemas import Race

def process_raw_weather(
    weather_path: Optional[str] = None,
    date_col: str = "date",
    weather_col: str = "weather",
    rain_col: str = "rain",
    save_to : Optional[str] = "data/processed/race_weather.csv",
) -> pd.DataFrame:
    """
    Read raw race weather CSV (no raceId expected), normalize dates, create a
    binary `rain` feature and persist a minimal (date, rain) table.
    Logic mirrors the notebook: rain if weather in
    ['Rain','Changeable','Very changeable'] (case-insensitive).
    """
    project_root = Path(__file__).resolve().parents[2]
    # resolve path
    if weather_path is None:
        candidate = project_root / "data" / "raw" / "race_weather.csv"
        weather_path = str(candidate) if candidate.exists() else None

    if weather_path is None:
        return pd.DataFrame(columns=[date_col, rain_col])

    wp = Path(weather_path)
    if not wp.is_absolute():
        wp = project_root / wp

    dfw = pd.read_csv(str(wp), dtype=str)

    # # normalize date column
    # if date_col in dfw.columns:
    #     dfw = normalize_race_dates(dfw, date_col)
    #     dfw[date_col] = dfw[date_col].dt.date
    if date_col in dfw.columns:
        dfw[date_col] = pd.to_datetime(dfw[date_col], format="%Y-%m-%d", errors="coerce").dt.date

    # create rain feature (handle NaN and common variants)
    def _is_rain(x):
        if pd.isna(x):
            return 0
        v = str(x).strip().lower()
        return 1 if v in {"rain", "rainy", "changeable", "very changeable"} else 0

    if weather_col in dfw.columns:
        dfw[rain_col] = dfw[weather_col].apply(_is_rain)
    else:
        dfw[rain_col] = 0

    # keep only minimal columns (date, rain)
    keep_cols = [c for c in [date_col, rain_col] if c in dfw.columns]
    out = dfw[keep_cols].drop_duplicates(subset=[date_col])

    if save_to:
        save_path = Path(save_to)
        if not save_path.is_absolute():
            save_path = project_root / save_path
        os.makedirs(save_path.parent, exist_ok=True)
        out.to_csv(str(save_path), index=False)

    return out

def load_races(
    path: Optional[str] = None,
    weather_path: Optional[str] = None,
    date_col: str = "date",
    save_to: Optional[str] = "data/processed/races.csv",
    metadata_path : Optional[str] = "data/processed/race_weather.csv",
) -> List[Race]:
    """
    Read raw races CSV, process raw weather (no raceId) into a small date->rain
    table, merge rain into races by date, persist processed races (keeping
    'rain' as a feature) and return Race DTOs.
    """
    project_root = Path(__file__).resolve().parents[2]
    # resolve races path
    if path is None:
        races_path = project_root / "data" / "raw" / "races.csv"
    else:
        races_path = Path(path)
        if not races_path.is_absolute():
            races_path = project_root / races_path

    df = pd.read_csv(str(races_path), dtype=str)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce").dt.date

    # process raw weather into small table with rain feature
    weather_meta = process_raw_weather(weather_path=weather_path, date_col=date_col,
                                       weather_col="weather", rain_col="rain", save_to=metadata_path)

    # merge rain into races by date (if available)
    if not weather_meta.empty and date_col in df.columns and date_col in weather_meta.columns:
        # merge and fill missing rain with 0
        df = df.merge(weather_meta[[date_col, "rain"]].drop_duplicates(subset=[date_col]),
                      how="left", on=date_col)
        df["rain"] = df["rain"].fillna(0).astype(int)

    # normalize numeric columns where present
    for col in ("raceId", "year", "round", "circuitId"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # keep date as python date in returned DTOs
    if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = df[date_col].dt.date

    # persist cleaned races (keep 'rain' feature; remove any raw 'weather' column)
    df_to_persist = df.copy()
    if "weather" in df_to_persist.columns:
        df_to_persist = df_to_persist.drop(columns=["weather"])
    if save_to:
        save_path = Path(save_to)
        if not save_path.is_absolute():
            save_path = project_root / save_path
        os.makedirs(save_path.parent, exist_ok=True)
        df_to_persist.to_csv(str(save_path), index=False)

    # build DTO list (ensure rain is int)
    races: List[Race] = []
    for _, row in df.iterrows():
        obj = {
            "raceId": None if pd.isna(row.get("raceId")) else int(row.get("raceId")),
            "year": None if pd.isna(row.get("year")) else int(row.get("year")),
            "round": None if pd.isna(row.get("round")) else int(row.get("round")),
            "circuitId": None if pd.isna(row.get("circuitId")) else int(row.get("circuitId")),
            "name": row.get("name"),
            "date": row.get(date_col) if date_col in row and not pd.isna(row.get(date_col)) else None,
            "time": row.get("time"),
            "url": row.get("url"),
            # raw weather not kept on races DTO; expose rain as feature if present
            "weather": None,
            "rain": int(row.get("rain")) if "rain" in row and not pd.isna(row.get("rain")) else 0,
        }
        races.append(Race.model_validate(obj))

    return races

def process_countries(
        path: Optional[str] = None,
        save_to: Optional[str] = "data/processed/countries.csv",) -> pd.DataFrame:
    """
    Load countries CSV and normalize for joins:
      - keep first nationality token (before comma)
      - keep first en_short_name token
      - map known long names to short codes (UK, UAE, Korea, Russia, USA)
      - drop problematic entries (United States Minor Outlying Islands)
    Returns a cleaned countries DataFrame (contains alpha_3_code and other columns).
    """
    project_root = Path(__file__).resolve().parents[2]
    # resolve countries path
    if path is None:
        countries_path = project_root / "data" / "raw" / "countries.csv"
    else:
        countries_path = Path(path)
        if not countries_path.is_absolute():
            countries_path = project_root / countries_path

    df = pd.read_csv(countries_path, dtype=str)

    # defensive defaults
    df['nationality'] = df.get('nationality', '').fillna('').astype(str)
    df['en_short_name'] = df.get('en_short_name', '').fillna('').astype(str)

    # keep first value before comma
    df['nationality'] = df['nationality'].apply(lambda x: x.split(',')[0].strip())
    df['en_short_name'] = df['en_short_name'].apply(lambda x: x.split(',')[0].strip())

    # normalize a few special long names to short/common tokens (matches notebook)
    mapping = {
        'United Kingdom of Great Britain and Northern Ireland': 'UK',
        'United Arab Emirates': 'UAE',
        'Korea (Republic of)': 'Korea',
        'Russian Federation': 'Russia',
        'United States of America': 'USA'
    }
    df['en_short_name'] = df['en_short_name'].apply(lambda x: mapping.get(x, x))

    # drop problematic duplicates / entries as in notebook
    df = df[~df['en_short_name'].isin(['United States Minor Outlying Islands'])].copy()

    # Ensure column types stable
    df = df.reset_index(drop=True)

    if save_to:
        save_path = Path(save_to)
        if not save_path.is_absolute():
            save_path = project_root / save_path
        os.makedirs(save_path.parent, exist_ok=True)
        df.to_csv(str(save_path), index=False)

    return df

def merge_country_codes(df: pd.DataFrame,
                        countries_df: pd.DataFrame,
                        left_on: str,
                        right_on: str = "nationality",
                        alpha3_col: str = "alpha_3_code",
                        result_col: Optional[str] = None) -> pd.DataFrame:
    """
    Merge `countries_df` into `df` matching left_on -> right_on and
    expose the alpha3 code (or other alpha3_col) as `result_col` (defaults to alpha3_col).
    Drops the helper join columns added from countries_df after the merge.
    """
    result_col = result_col or alpha3_col
    merged = df.merge(
        countries_df,
        how='left',
        left_on=left_on,
        right_on=right_on,
        suffixes=('', '_country')
    )
    # copy alpha3 into desired column name
    merged[result_col] = merged.get(alpha3_col)
    # drop the extra columns from countries df that we don't need
    drop_cols = [c for c in ['num_code', 'alpha_2_code', 'en_short_name', 'nationality'] if c in merged.columns]
    merged = merged.drop(columns=drop_cols, errors='ignore')
    # if original left_on should be removed (not always), caller can drop it
    return merged

def _load_countries_preferred(countries_path: str = "data/processed/countries.csv") -> pd.DataFrame:
    """
    Load processed countries CSV if present, otherwise call process_countries(...)
    Returns a cleaned countries DataFrame ready for merges.
    """
    project_root = Path(__file__).resolve().parents[2]
    countries_file = Path(countries_path)
    if not countries_file.is_absolute():
        countries_file = project_root / countries_file

    if countries_file.exists():
        countries = pd.read_csv(countries_file, dtype=str)
    else:
        countries = process_countries(path=None)

    return countries

def build_driver_country_table(
    drivers_path: Optional[str] = None,
    countries_path: str = "data/processed/countries.csv",
    race_path: str = "data/raw/races.csv",
    result_path: str = "data/raw/results.csv",
    save_to: str = "data/processed/drivers.csv",
):
    """
    Build and persist unique driver table with alpha-3 nationality.

    - If drivers_path is provided and is a path, the CSV at that path is loaded.
    - If drivers_path is None, the function loads raw drivers CSV from
      data/raw/drivers.csv.
    - Merges alpha-3 codes using _load_countries_preferred / process_countries
      and merge_country_codes(...) helpers.
    """
    project_root = Path(__file__).resolve().parents[2]
    # Resolve drivers CSV
    if drivers_path is None:
        raw_path = project_root / "data" / "raw" / "drivers.csv"
    else:
        raw_path = Path(drivers_path)
        if not raw_path.is_absolute():
            raw_path = project_root / raw_path

    df = pd.read_csv(raw_path, dtype=str)

    # normalize to expected columns
    if "driver" not in df.columns:
        if {"forename", "surname"}.issubset(df.columns):
            df["driver"] = df["forename"].fillna("") + " " + df["surname"].fillna("")
            df["driver"] = df["driver"].str.strip()
        elif "driverRef" in df.columns:
            df["driver"] = df["driverRef"]
        else:
            df["driver"] = df.iloc[:, 0].astype(str)

    # pick dob column if present
    dob_col = next((c for c in ("driver_date_of_birth", "dob", "date_of_birth") if c in df.columns), None)
    if dob_col and dob_col != "driver_date_of_birth":
        df = df.rename(columns={dob_col: "driver_date_of_birth"})

    # pick nationality column
    nat_col = next((c for c in ("nationality", "driver_nationality") if c in df.columns), None)
    if nat_col and nat_col != "driver_nationality":
        df = df.rename(columns={nat_col: "driver_nationality"})
    elif nat_col is None:
        df["driver_nationality"] = None

    #   change value of certain nationalities to match countries.csv

    df['driver_nationality'] = df['driver_nationality'].str.strip()
    df['driver_nationality'] = df['driver_nationality'].replace(
        {'Rhodesian': 'Zimbabwean',
        'American-Italian': 'American',
         'Argentine-Italian': 'Argentine',
         'East German': 'German',
         'West German': 'German',
         'Argentinian' : 'Argentine',})

    countries = _load_countries_preferred(countries_path)
    merged = merge_country_codes(df, countries, left_on="driver_nationality", result_col="driver_nationality")

    # User other dataset to get drivers first race date
    project_root = Path(__file__).resolve().parents[2]
    # Resolve drivers CSV

    if race_path is None:
        raw_race_path = project_root / "data" / "raw" / "races.csv"
    else:
        raw_race_path = Path(race_path)
        if not raw_race_path.is_absolute():
            raw_race_path = project_root / raw_race_path

    races = pd.read_csv(Path(raw_race_path), dtype=str)

    if result_path is None:
        raw_result_path = project_root / "data" / "raw" / "results.csv"
    else:
        raw_result_path = Path(result_path)
        if not raw_result_path.is_absolute():
            raw_result_path = project_root / raw_result_path

    results = pd.read_csv(Path(raw_result_path), dtype=str)
    drivers = merged.copy()
    df1 = pd.merge(races, results, how="left", on=["raceId"], suffixes=("_race", "_result"))
    df2 = pd.merge(df1, drivers, how="left", on=["driverId"], suffixes=("", "_driver"))
    first_race_dates = df2.groupby("driverId")["date"].min().reset_index()
    first_race_dates = first_race_dates.rename(columns={"date": "first_race_date"})
    data = df2.merge(first_race_dates, on="driverId", how="left")

    # merge first race date into drivers
    merged_drivers = pd.merge(merged, data, how="left", on=["driverId"], suffixes=("", "_y"))


    out = merged_drivers[["driver","driverRef", "driver_date_of_birth", "driver_nationality", 'first_race_date']].drop_duplicates().reset_index(drop=True)

    if save_to:
        save_path = Path(save_to)
        if not save_path.is_absolute():
            save_path = project_root / save_path
        os.makedirs(save_path.parent, exist_ok=True)
        out.to_csv(str(save_path), index=False)

def build_constructor_country_table(
    constructors_path: Optional[str] = None,
    countries_path: str = "data/processed/countries.csv",
    save_to: str = "data/processed/constructors.csv",
):
    """
    Build and persist unique constructor table with alpha-3 nationality.

    - If constructors_path is None, loads data/raw/constructors.csv.
    """
    project_root = Path(__file__).resolve().parents[2]
    # Resolve constructors CSV
    if constructors_path is None:
        raw_path = project_root / "data" / "raw" / "constructors.csv"
    else:
        raw_path = Path(constructors_path)
        if not raw_path.is_absolute():
            raw_path = project_root / raw_path

    df = pd.read_csv(raw_path, dtype=str)

    if "constructor" not in df.columns:
        if "name" in df.columns:
            df["constructor"] = df["name"]
        elif "constructorRef" in df.columns:
            df["constructor"] = df["constructorRef"]
        else:
            df["constructor"] = df.iloc[:, 0].astype(str)

    nat_col = next((c for c in ("nationality", "constructor_nationality") if c in df.columns), None)
    if nat_col and nat_col != "constructor_nationality":
        df = df.rename(columns={nat_col: "constructor_nationality"})
    elif nat_col is None:
        df["constructor_nationality"] = None

    df['constructor_nationality'] = df['constructor_nationality'].str.strip()
    df['constructor_nationality'] = df['constructor_nationality'].replace(
        {'Rhodesian': 'Zimbabwean',
        'American-Italian': 'American',
         'Argentine-Italian': 'Argentine',
         'East German': 'German',
         'West German': 'German',
         'Argentinian' : 'Argentine',})

    countries = _load_countries_preferred(countries_path)
    merged = merge_country_codes(df, countries, left_on="constructor_nationality", result_col="constructor_nationality")

    out = merged[["constructor", "constructorRef", "constructor_nationality"]].drop_duplicates().reset_index(drop=True)

    if save_to:
        save_path = Path(save_to)
        if not save_path.is_absolute():
            save_path = project_root / save_path
        os.makedirs(save_path.parent, exist_ok=True)
        out.to_csv(str(save_path), index=False)

def build_circuit_country_table(
    circuits_path: Optional[str] = None,
    countries_path: str = "data/processed/countries.csv",
    save_to: str = "data/processed/circuits.csv",
):
    """
    Build and persist unique circuit table with alpha-3 nationality.

    - If circuits_path is None, loads data/raw/circuits.csv.
    """
    project_root = Path(__file__).resolve().parents[2]
    # Resolve circuits CSV
    if circuits_path is None:
        raw_path = project_root / "data" / "raw" / "circuits.csv"
    else:
        raw_path = Path(circuits_path)
        if not raw_path.is_absolute():
            raw_path = project_root / raw_path

    df = pd.read_csv(raw_path, dtype=str)

    if "circuit" not in df.columns:
        if "name" in df.columns:
            df["circuit"] = df["name"]
        elif "circuitRef" in df.columns:
            df["circuit"] = df["circuitRef"]
        else:
            df["circuit"] = df.iloc[:, 0].astype(str)

    if "type" in df.columns and "type_circuit" not in df.columns:
        df = df.rename(columns={"type": "type_circuit"})
    elif "type_circuit" not in df.columns:
        df["type_circuit"] = None

    country_col = next((c for c in ("country", "country_circuit", "location") if c in df.columns), None)
    if country_col and country_col != "country_circuit":
        df = df.rename(columns={country_col: "country_circuit"})
    elif country_col is None:
        df["country_circuit"] = None

    countries = _load_countries_preferred(countries_path)
    merged = merge_country_codes(df, countries, left_on="country_circuit", right_on="en_short_name", result_col="circuit_nationality")

    out = merged[["circuit","circuitRef", "type_circuit", "circuit_nationality"]].drop_duplicates().reset_index(drop=True)

    if save_to:
        save_path = Path(save_to)
        if not save_path.is_absolute():
            save_path = project_root / save_path
        os.makedirs(save_path.parent, exist_ok=True)
        out.to_csv(str(save_path), index=False)
