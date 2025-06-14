

import logging
from lib import yaml
from lib.load_models import load_models

logger = logging.getLogger(__name__)

storage = {
    "models": {},
    "token_map": {}
}
    
def load_storage():
    """
    Load the in-memory storage with models and token map.
    This function is called at the start of the application.
    """
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