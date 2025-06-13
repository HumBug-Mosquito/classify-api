from pydantic import BaseModel


class AudioSegmentEventDetectionResult(BaseModel): 
    """
    Class to represent an audio segment passed through event detection.
    """
    
    species: str # Mosquito or Hornet
    start: float
    end: float
    probability: float
    classification_interval: float # seconds
    
def cast_to_audio_segments(predictions: list[float], prediction_interval: float = 1.0 ) -> list[AudioSegmentEventDetectionResult]:
    results: list[AudioSegmentEventDetectionResult] = []
    
    for i, prediction in enumerate(predictions):
        result = AudioSegmentEventDetectionResult(
            start=i * prediction_interval,
            end=(i + 1) * prediction_interval,
            probability=prediction,
            classification_interval=prediction_interval,
            species="mosquito",
        )
        results.append(result)

    return results
        

class EventDetectionResults(BaseModel):
    """
    Class to represent the results of an event detection model.
    """
    
    raw: list[float]
    species: str # Mosquito or Hornet
    segments: list[AudioSegmentEventDetectionResult]
    model: str
    time_taken: float