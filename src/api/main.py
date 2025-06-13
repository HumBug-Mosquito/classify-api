import asyncio
import base64
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import fastapi
import numpy as np
import redis.asyncio as aioredis
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from lib import yaml
from lib.load_models import load_models
from lib.prediction import run_prediction
from models.event_detection.med_1.main import Med1
from src.api.dto import PredictionResult
from src.lib.model import SupportedModel
from src.lib.utils import get_audio_from_file, get_duration, resample_audio
from src.workers.tasks import process_audio, storage

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # --- In-memory Storage ---
    storage["models"] = load_models()

    # --- Load and process authorized clients ---
    clients_config = yaml.load_config().get("authorized_clients", {})

    token_to_client_map = {}
    for client_key, client_data in clients_config.items():
        # NEW: Check if the client is enabled before processing.
        if not client_data.get("enabled", False):
            logger.info(f"Client '{client_key}' is disabled. Skipping.")
            continue

        token = client_data.get("token")
        if not token:
            logger.warning(f"Enabled client '{client_key}' is missing a token. Skipping.")
            continue
        if token in token_to_client_map:
            logger.warning(f"Duplicate token '{token}' found for client '{client_key}'. Overwriting.")
        
        token_to_client_map[token] = client_data

    storage["token_map"] = token_to_client_map
    logger.info(f"Loaded and processed {len(storage.get('token_map', {}))} enabled clients.")

    yield
    storage.clear()
    logger.info("Cleared in-memory storage.")

# --- Redis client ---
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

app = fastapi.FastAPI(
    title="Humbug Classification Service",
    lifespan=lifespan,
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
   allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_model_instance(model_name: str) -> SupportedModel:
    model: SupportedModel | None  = storage["models"].get(model_name)
    if model is None:
        raise fastapi.exceptions.HTTPException(
            status_code=400, detail=f"Model '{model_name}' not found."
        )
    return model


# --- Authorization ---

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_current_client(api_key: str = fastapi.Security(api_key_header)):
    """
    Dependency to get the current client from the API key.
    Uses the pre-computed token map for fast lookups.
    """
    if not api_key:
        raise fastapi.HTTPException(status_code=401, detail="API Key missing")

    token_map = storage.get("token_map", {})
    client_data = token_map.get(api_key)

    if not client_data:
        logger.warning(f"Invalid API Key: {api_key}")
        raise fastapi.HTTPException(status_code=401, detail="Invalid API Key")
    return client_data

def require_scopes(required_scopes: list[str]):
    async def scope_checker(client: dict = fastapi.Depends(get_current_client)):
        client_scopes = set(client.get("scopes", []))
        for scope in required_scopes:
            if scope not in client_scopes:
                raise fastapi.HTTPException(
                    status_code=403,
                    detail=f"Permission denied. Required scope: '{scope}'"
                )
        return client
    return scope_checker

# --- Endpoints ---

@app.get(
    "/health",
    summary="Health check endpoint",
    tags=["Health"],
)
async def health():
    try: await redis_client.ping(); return {"status": "ok"}
    except: raise fastapi.exceptions.HTTPException(503, "Redis unavailable")


@app.post(
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
    event_detection_model = get_model_instance(event_detection_model_name)
    species_detection_model = get_model_instance(species_detection_model_name) if species_detection_model_name else None

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


@app.post(
    "/recording/{recording_id}",
    summary="Provide an ID of a recording in the database to classify",
    tags=["Recording"],
)
def recording(recording_id: str):
    # Get the recording form the database, get the sample rate
    pass


@app.get(
    "/models",
    summary="Get a list of the available models",
    tags=["Model"],
)
def available_models(request: fastapi.Request):
    models = [supported_model.to_dict() for _, supported_model in storage["models"].items()]
    return models


@app.websocket("/ws/stream")
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
            process_audio(session_id, seq, chunk, ev_name, sp_name)
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