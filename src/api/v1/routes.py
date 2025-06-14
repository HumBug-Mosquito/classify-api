import asyncio
import json
import logging
import uuid
from typing import Optional

import fastapi
from fastapi import APIRouter

from src.api.dto import PredictionResult
from src.lib.prediction import run_prediction
from src.lib.utils import get_audio_from_file
from src.workers.tasks import process_audio
from src.api.core.redis import redis_client
from src.api.core.security import require_scopes, get_model_instance
from src.api.core.in_memory_storage import storage

logger = logging.getLogger(__name__)

# All routes in this file will be prefixed with /v1
v1_router = APIRouter()

@v1_router.get(
    "/health",
    summary="Health check endpoint",
    tags=["Health"],
)
async def health():
    try: await redis_client.ping(); return {"status": "ok"}
    except: raise fastapi.exceptions.HTTPException(503, "Redis unavailable")


@v1_router.post(
    "/file",
    summary="Predict presence in a .wav file",
    tags=["File"],
    response_model=PredictionResult,
)
async def file(
    audio_file: fastapi.UploadFile = fastapi.File(..., description="The audio file to classify (e.g., .wav)"),
    event_detection_model_name: str = fastapi.Form(..., alias="event_detection_model"),
    species_detection_model_name: Optional[str] = fastapi.Form(None, alias="species_detection_model"),
     # Scopes can be adjusted based on the new schema, e.g. "predict:med"
    client: dict = fastapi.Depends(require_scopes(["predict:med"])),
):
    event_detection_model = get_model_instance( event_detection_model_name) 
    species_detection_model = get_model_instance( species_detection_model_name) if species_detection_model_name else None 

    logger.info(f"Received file: {audio_file.filename}, Event Model: {event_detection_model_name}, Species Model: {species_detection_model_name}")

    audio, sample_rate = get_audio_from_file(audio_file.file)

    try:
        event_results, species_results = run_prediction(
            audio, sample_rate, event_detection_model, species_detection_model
        )
        message = (
            "Finished detecting events and species in audio."
            if species_detection_model
            else "Finished detecting events in audio."
        )
        return PredictionResult(
            message=message,
            event_detection_result=event_results,
            species_detection_result=species_results,
        )
    
    except ValueError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))


@v1_router.post(
    "/recording/{recording_id}",
    summary="Provide an ID of a recording in the database to classify",
    tags=["Recording"],
)
def recording(recording_id: str):
    # Get the recording form the database, get the sample rate
    pass


@v1_router.get(
    "/models",
    summary="Get a list of the available models",
    tags=["Model"],
)
def available_models(request: fastapi.Request):
    models = [supported_model.to_dict() for _, supported_model in storage["models"].items()]
    return models


@v1_router.websocket("/ws/stream")
async def websocket_stream(websocket: fastapi.WebSocket):
    await websocket.accept()

    ev_name = websocket.query_params.get("event_model")
    if ev_name is None:
        raise fastapi.exceptions.HTTPException(
            status_code=400, detail="Missing 'event_model' query parameter"
        )
    
    sp_name = websocket.query_params.get("species_model")

    session_id = websocket.query_params.get("session_id") or str(uuid.uuid4())
    await websocket.send_json({"session_id": session_id})

    pubsub = redis_client.pubsub()
    await pubsub.subscribe(f"results:{session_id}")

    async def recv_loop():
        seq = 0
        while True:
            chunk = await websocket.receive_bytes()
            process_audio.delay(session_id, seq, chunk, ev_name, sp_name)
            seq += 1

    async def send_loop():
        async for msg in pubsub.listen():
            if msg.get("type") != "message": continue
            await websocket.send_json(json.loads(msg["data"]))

    try:
        await asyncio.gather(recv_loop(), send_loop())
    except fastapi.WebSocketDisconnect:
        logger.info(f"Disconnected: {session_id}")
    finally:
        await pubsub.unsubscribe(f"results:{session_id}")
        await pubsub.close()