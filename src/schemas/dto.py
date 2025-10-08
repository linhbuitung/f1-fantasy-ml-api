from typing import Optional
from datetime import date
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

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

class PredictionItem(BaseModel):
    input: "MainRacePredictInput"  # preserve original DTO (nested)
    features: Dict[str, Any] = Field(..., description="Expanded features passed to the models")
    predicted_deviation_from_median: float = Field(..., description="Model predicted deviation (ms)")
    predicted_final_position: Optional[int] = Field(
        None, description="Derived rank in race (if computed from batch/race context)"
    )

class MainRacePredictResponse(BaseModel):
    predictions: List[PredictionItem] = Field(..., description="List of prediction results")
    model_meta: Dict[str, Any] = Field(..., description="Model metadata / provenance")