
from typing import Optional

from pydantic import BaseModel, Field

from src.lib.event_detection_result import EventDetectionResults
from src.lib.species_detection_result import SpeciesDetectionResults

    
class PredictionResult(BaseModel):
    message: str
    event_detection_result: EventDetectionResults
    species_detection_result: Optional[SpeciesDetectionResults]

