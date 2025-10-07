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