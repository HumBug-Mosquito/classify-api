import base64
import io
import json
import logging
import os
import time
from typing import Optional

import numpy as np
import redis
from celery import Task

from models.event_detection.med_1.main import Med1
from src.lib.event_detection_result import EventDetectionResults
from src.lib.model import SupportedModel
from src.lib.species_detection_result import SpeciesDetectionResults
from src.lib.utils import get_audio_from_file, get_duration, resample_audio
from src.workers.celery_app import celery_app

# --- In-memory Storage ---
storage = {
    "models": {},
    "authorized_api_keys": list[str]
}

# --- Logging & Redis client ---
logger = logging.getLogger(__name__)
redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://redis:6379/0"),
    decode_responses=True,
)

def load_models():
    if storage["models"]:
        return
    logger.info("Loading models in worker...")
    med = Med1(); med.load()
    storage["models"]["med-1"] = med
    # (Optional) load species models here

class BaseTaskWithModels(Task):
    def __call__(self, *args, **kwargs):
        load_models()
        return self.run(*args, **kwargs)

@celery_app.task(base=BaseTaskWithModels, name="process_audio")
def process_audio(
    session_id: str,
    sequence: int,
    audio: bytes,
    event_model_name: str,
    species_model_name: Optional[str] = None,
) -> dict:
    """
    Decode audio, run prediction, publish via Redis, cache latest result.
    """
    # Decode frame
    audio_file = io.BytesIO(audio)
    audio_array, sample_rate = get_audio_from_file(audio_file)

    # Fetch models
    event_model: SupportedModel = storage["models"].get(event_model_name)
    if not event_model:
        raise ValueError(f"Event model '{event_model_name}' not found")
    species_model: Optional[SupportedModel] = None
    if species_model_name:
        species_model = storage["models"].get(species_model_name)

    # Ensure duration
    duration = get_duration(audio_array, sample_rate)
    if duration < event_model.requirements.min_duration:
        raise ValueError("Audio chunk too short for model requirements")

    # Resample if needed
    if sample_rate != event_model.requirements.sample_rate:
        audio_array = resample_audio(
            audio_array,
            original_sample_rate=sample_rate,
            target_sample_rate=event_model.requirements.sample_rate,
        )

    # Run inference
    event_results: EventDetectionResults = event_model.predict(audio_array)
    species_results: Optional[SpeciesDetectionResults] = None
    # (Optional) run species detection

    # Build payload
    payload = {
        "session_id": session_id,
        "sequence": sequence,
        "result": {
            "timestamp_ms": int(time.time() * 1000),
            "message": f"Batch {sequence} - processed",
            "event_detection": event_results.model_dump(),
            "species_detection": species_results.model_dump() if species_results else None,
        }
    }
    message = json.dumps(payload)

    # Publish and cache
    redis_client.publish(f"results:{session_id}", message)
    redis_client.set(f"latest_result:{session_id}", message)
    redis_client.expire(f"latest_result:{session_id}", 3600)

    return payload