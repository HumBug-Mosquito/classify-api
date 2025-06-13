# src/lib/config.py
import logging
import os
import yaml
from pathlib import Path

logger = logging.getLogger("src.lib.yaml")

def load_config() -> dict[str, any]: # type: ignore
    """Loads the application configuration from config.yml."""
    config_path = Path(__file__).parent.parent.parent / "config.yml"
    
    logger.info(f"Loading configuration from: {config_path}")
    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_path}")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

