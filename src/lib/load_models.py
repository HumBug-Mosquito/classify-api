# src/lib/model_loader.py
import logging
from typing import Dict

from models.event_detection.med_1.main import Med1
from src.lib.model import SupportedModel

logger = logging.getLogger(__name__)

def load_models() -> Dict[str, SupportedModel]:
    """
    Loads all supported models into a dictionary.

    Returns:
        A dictionary of loaded models.
    """
    logger.info("Loading models...")
    med = Med1()
    med.load()

    models = {"med-1": med}
    logger.info(f"Loaded {len(models)} model(s).")
    return dict(models)