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

class MainRacePredictionItem(BaseModel):
    input: "MainRacePredictInput"  # preserve original DTO (nested)
    features: Dict[str, Any] = Field(..., description="Expanded features passed to the models")
    predicted_deviation_from_median: float = Field(..., description="Model predicted deviation (ms)")
    predicted_final_position: Optional[int] = Field(
        None, description="Derived rank in race (if computed from batch/race context)"
    )

class MainRacePredictResponse(BaseModel):
    predictions: List[MainRacePredictionItem] = Field(..., description="List of prediction results")
    model_meta: Dict[str, Any] = Field(..., description="Model metadata / provenance")

class QualifyingPredictInput(BaseModel):
    constructor: str
    circuit: str
    driver: str
    race_date: date

class QualifyingPredictionItem(BaseModel):
    input: "QualifyingPredictInput"  # preserve original DTO (nested)
    features: Dict[str, Any] = Field(..., description="Expanded features passed to the models")
    predicted_deviation_from_median: float = Field(..., description="Model predicted deviation (ms)")
    predicted_final_position: Optional[int] = Field(
        None, description="Derived rank in race (if computed from batch/race context)"
    )

class QualifyingPredictResponse(BaseModel):
    predictions: List[QualifyingPredictionItem] = Field(..., description="List of prediction results")
    model_meta: Dict[str, Any] = Field(..., description="Model metadata / provenance")

class StatusPredictInput(BaseModel):
    qualification_position: int
    constructor: str
    circuit: str
    driver: str
    race_date: date

# Replace per-record prediction item with aggregate result per race
class StatusPredictionItem(BaseModel):
    input: "StatusPredictInput"
    features: Dict[str, Any] = Field(..., description="Expanded features passed to the models")
    dnf_percentage: float = Field(..., description="DNF percentage (0-100) for this race")

class StatusPredictResponse(BaseModel):
    percentages: List[StatusPredictionItem] = Field(..., description="Per-race DNF percentages for requested inputs")
    # keep model_meta for parity with other endpoints but optional here
    model_meta: Optional[Dict[str, Any]] = Field(None, description="Optional metadata/provenance")