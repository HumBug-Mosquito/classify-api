
from common.dto import EventDetectionPrediction, EventDetectionResults, HighestConfidenceSpecies, SpeciesDetectionPrediction, SpeciesDetectionResults


def create_dto_from_med_predictions(
        med_predictions: list[float], 
        model_name:str,
        vad_mask: list[bool] | None = None,
):
    """
    Create a DTO from the MED predictions.

    Args:
        med_predictions (list): List of MED predictions.
        model_name (str): Name of the model used for prediction.

    Returns:
        EventDetectionResults: DTO containing the event detection results.
    """
    predictions: list[EventDetectionPrediction] = []
    for start_time, confidence in enumerate(med_predictions):
        end_time = start_time + 1  # Assuming each prediction corresponds to a 1-second interval
        predictions.append(EventDetectionPrediction(
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            contains_voice=vad_mask[start_time] if vad_mask else False
        ))

    return EventDetectionResults(
        model_name=model_name,
        predictions=predictions,
        raw=med_predictions,
    )

def create_dto_from_species_predictions(
        species_predictions: list[dict[str, float]], 
        model_name: str
):
    """
    Create a DTO from the species predictions.

    Args:
        species_predictions (list): List of species predictions, each represented as a dictionary with species names as keys and confidence scores as values.
        model_name (str): Name of the model used for prediction.
    Returns:
        SpeciesDetectionResults: DTO containing the species detection results.
    """
    predictions: list[SpeciesDetectionPrediction] = []
    for start_time, species_prediction in enumerate(species_predictions):
        # Get the species with the highest confidence score
        species =  max(species_prediction.keys(), key=species_prediction.get) # type: ignore
        highest_confidence = HighestConfidenceSpecies(species=species, confidence=species_prediction[species])
        predictions.append(SpeciesDetectionPrediction(
            start_time=start_time,
            end_time=start_time + 1,    # Assuming end time is 1 second after start time
            highest_confidence=highest_confidence,
            raw=species_prediction
        ))

    return SpeciesDetectionResults(
        model_name=model_name,
        predictions=predictions
    )