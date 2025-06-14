# --- Authorization ---
import logging
import fastapi
import fastapi.security

from src.lib.model import SupportedModel
from src.api.core.in_memory_storage import storage

logger = logging.getLogger(__name__)
api_key_header = fastapi.security.APIKeyHeader(name="X-API-Key", auto_error=False)

def get_model_instance( model_name: str) -> SupportedModel:
    model: SupportedModel | None  = storage.get("models", {}).get(model_name)
    if model is None:
        raise fastapi.exceptions.HTTPException(
            status_code=400, detail=f"Model '{model_name}' not found."
        )
    return model

async def get_current_client(api_key: str = fastapi.Security(api_key_header)):
    """
    Dependency to get the current client from the API key.
    Uses the pre-computed token map for fast lookups.
    """
    if not api_key:
        raise fastapi.HTTPException(status_code=401, detail="API Key missing")

    client_data = storage.get("token_map", {}).get(api_key)

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
