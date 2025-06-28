

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

# ------------- Event Detection -------------

class EventDetectionPrediction(BaseModel):
    start_time: float = Field(..., description="Start time of the detected event in seconds from the start of the audio.")
    end_time: float = Field(..., description="End time of the detected event in seconds from the start of the audio.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the event detection (0 to 1).")
    contains_voice: bool = Field(default=False, description="Indicates if the event contains voice activity (True/False).")

class EventDetectionResults(BaseModel):
    model_name: str = Field(..., description="Name of the model used for prediction.")
    predictions: list[EventDetectionPrediction] = Field(default_factory=list, description="List of event detection predictions.")
    raw: list[float] = Field(default_factory=list, description="Raw output of the model as a list of confidence scores.")

# ------------- Species Detection -------------
class HighestConfidenceSpecies(BaseModel):
    species: str = Field(..., description="Name of the species with the highest confidence score.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the species detection (0 to 1).")   

class SpeciesDetectionPrediction(BaseModel):
    start_time: float = Field(..., description="Start time of the detected event in seconds from the start of the audio.")
    end_time: float = Field(..., description="End time of the detected event in seconds from the start of the audio.")
    highest_confidence: HighestConfidenceSpecies = Field(..., description="Species with the highest confidence score.")
    raw: dict[str, float] = Field(default_factory=dict, description="Dictionary of species names and their confidence scores.")

class SpeciesDetectionResults(BaseModel):
    model_name: str = Field(..., description="Name of the model used for prediction.")
    predictions: list[SpeciesDetectionPrediction] = Field(default_factory=list, description="List of species detection predictions.")

# ------------- API Response Payload -------------

class APIResponsePayload(BaseModel):
    message: str = Field(..., description="Message describing the response, e.g., 'Prediction completed successfully'.")

# ------------- Prediction Response -------------

class PredictionPayload(APIResponsePayload):
    steps_completed: dict[str, str] = Field(default_factory=dict, description="List of steps completed in the prediction process.")
    event_detection: EventDetectionResults = Field(..., description="Results of the event detection model.")
    species_detection: Optional[SpeciesDetectionResults] = Field(default=None, description="Results of the species detection model.")

# ------------- Error Response -------------

class ErrorPayload(APIResponsePayload):
    code: int = Field(default=500, description="HTTP status code for the error response.")
    error: str = Field(..., description="Error message describing what went wrong.")

# ------------- API Response -------------

class ResponseMetadata(BaseModel):
    request_id: Optional[str] = Field(default=None, description="Unique identifier for the request, if available.")
    timestamp: str = Field(default_factory=datetime.now().isoformat, description="Timestamp when the response was generated.")
    params: dict = Field(default_factory=dict)

class APIResponse(BaseModel):
    type: str = Field(..., description="Type of the API response, e.g., 'error | success'.")
    metadata: ResponseMetadata = Field(..., description="Metadata about the response, including request ID and timestamp.")

class ErrorResponse(APIResponse):
    type: str = Field(default="error")
    payload: ErrorPayload = Field(..., description="Payload containing error details.")

class SuccessfulResponse(APIResponse):
    type: str = Field(default="success.file")
    payload: PredictionPayload = Field(..., description="Payload containing prediction results.")

# ------------- Streamed Response -------------

class StreamPredictionPayload(PredictionPayload):
    """
    Payload for streamed predictions, simply includes a chunk index so that the client can correlate responses with requests / order the responses.
    """
    chunk_index: int = Field(..., description="Index of the chunk in the streamed response.")

class SuccessfulStreamedResponse(APIResponse):
    type: str = Field(default="success.batch")
    payload: StreamPredictionPayload = Field(..., description="Payload containing streamed prediction results.")

class StreamFinished(APIResponse):
    type: str = Field(default="success.stream_finished")
    payload: APIResponsePayload = Field(..., description="Payload indicating that the stream has finished.")