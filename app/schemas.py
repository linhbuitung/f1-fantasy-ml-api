from typing import Optional
from datetime import date
from pydantic import BaseModel

class Race(BaseModel):
    raceId: Optional[int]
    year: Optional[int]
    round: Optional[int]
    circuitId: Optional[int]
    name: Optional[str]
    date: Optional[date]
    time: Optional[str]
    url: Optional[str]
    weather: Optional[str]

class MainRacePredictInput(BaseModel):
    qualification_position: int
    laps: int
    constructor: str
    circuit: str
    driver: str
    race_date: date
    rain: Optional[int] = 0