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
from src.api.core.in_memory_storage import load_storage, storage

from src.api.v1.routes import v1_router
# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    load_storage()
    yield
    storage.clear()
    logger.info("Cleared in-memory storage.")

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

app.include_router(router=v1_router, prefix="/v1")