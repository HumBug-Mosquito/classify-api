# src/lib/prediction.py
import logging
from typing import Optional

import numpy as np

from src.lib.event_detection_result import EventDetectionResults
from src.lib.model import SupportedModel
from src.lib.species_detection_result import SpeciesDetectionResults
from src.lib.utils import get_duration, resample_audio

logger = logging.getLogger(__name__)


def run_prediction(
    audio: np.ndarray,
    sample_rate: int,
    event_model: SupportedModel,
    species_model: Optional[SupportedModel] = None,
) -> tuple[EventDetectionResults, Optional[SpeciesDetectionResults]]:
    """
    Runs event and species detection on an audio array.

    Args:
        audio: The audio data.
        sample_rate: The sample rate of the audio.
        event_model: The event detection model to use.
        species_model: The species detection model to use (optional).

    Returns:
        A tuple containing the event detection results and species detection results.
    """
    duration = get_duration(audio, sample_rate)
    if duration < event_model.requirements.min_duration:
        raise ValueError("Audio chunk too short for model requirements")

    if sample_rate != event_model.requirements.sample_rate:
        audio = resample_audio(
            audio,
            original_sample_rate=sample_rate,
            target_sample_rate=event_model.requirements.sample_rate,
        )

    event_results = event_model.predict(audio)
    species_results = None
    if species_model:
        # Placeholder for species detection logic
        logger.info(f"Running species detection with {species_model.name}")
        # species_results = species_model.predict(audio)

    return event_results, species_results