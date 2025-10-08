from pathlib import Path
import pandas as pd
from src.preprocess import process_raw_weather, load_races

def _write_csv(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

def test_process_raw_weather_creates_rain_and_persists(tmp_path):
    weather_csv = tmp_path / "raw_weather.csv"
    weather_content = "date,weather\n2020-07-05,Rain\n2020-07-06,Sunny\n2020-07-07,Very changeable\n"
    _write_csv(weather_csv, weather_content)

    out_path = tmp_path / "processed_race_weather.csv"
    df = process_raw_weather(weather_path=str(weather_csv), date_col="date", weather_col="weather", rain_col="rain", save_to=str(out_path))
    # returned frame contains date and rain
    assert "rain" in df.columns
    assert set(df["rain"].tolist()) <= {0, 1}
    # persisted file exists
    assert out_path.exists()
    persisted = pd.read_csv(out_path)
    assert "rain" in persisted.columns
    assert "date" in persisted.columns

def test_load_races_merges_rain_and_persists(tmp_path):
    # races CSV (minimal)
    races_csv = tmp_path / "races.csv"
    races_content = "raceId,year,round,circuitId,name,date,time,url\n1,2020,1,1,Test GP,2020-07-05,13:00:00,http://example.com\n2,2020,2,1,Test GP 2,2020-07-06,13:00:00,http://example.com\n"
    _write_csv(races_csv, races_content)

    # weather CSV matching dates
    weather_csv = tmp_path / "race_weather.csv"
    weather_content = "date,weather\n2020-07-05,Rain\n2020-07-06,Sunny\n"
    _write_csv(weather_csv, weather_content)

    processed_races = tmp_path / "processed_races.csv"
    metadata = tmp_path / "processed_race_weather.csv"

    races = load_races(path=str(races_csv), weather_path=str(weather_csv),
                       date_col="date",
                       save_to=str(processed_races),
                       metadata_path=str(metadata))

    # function returns list of DTOs
    assert isinstance(races, list)
    assert len(races) == 2

    # processed races file was written and contains rain column
    assert processed_races.exists()
    df_proc = pd.read_csv(processed_races)
    assert "rain" in df_proc.columns

    # metadata persisted and contains date+rain
    assert metadata.exists()
    df_meta = pd.read_csv(metadata)
    assert set(df_meta.columns) >= {"date", "rain"}