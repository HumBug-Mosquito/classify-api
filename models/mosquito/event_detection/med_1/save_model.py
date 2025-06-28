import os
import logging
import bentoml
import torch
from typing import Dict, Any, List
from types import ModuleType

# To make the import below work, this script must be run as a module
# from the root of your repository, e.g., `python -m med_1.save_bento`
import models.mosquito.event_detection.med_1.framework as framework
from models.mosquito.event_detection.med_1.framework.models import load_pickle, MTRCNN_pad_half


# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_NAME = "med-general-1"

def load_normalization_data(norm_file_path: str) -> Dict[str, Any]:
    """Loads the normalization pickle file."""
    logging.info(f"Loading normalization parameters from: {norm_file_path}")
    if not os.path.exists(norm_file_path):
        logging.error(f"Normalization file not found at {norm_file_path}")
        raise FileNotFoundError(f"File not found: {norm_file_path}")
    return load_pickle(norm_file_path)

def create_and_prepare_model(norm_data: Dict[str, Any], weights_path: str) -> MTRCNN_pad_half:
    """Instantiates the model, loads weights, and sets it to evaluation mode."""
    logging.info("Creating model instance.")
    model = MTRCNN_pad_half(
        class_num=1,
        dropout=0.2,
        MC_dropout=True,
        batchnormal=True,
        mean=norm_data['mean'],
        std=norm_data['std']
    )

    logging.info(f"Loading pre-trained weights from: {weights_path}")
    if not os.path.exists(weights_path):
        logging.error(f"Model weights file not found at {weights_path}")
        raise FileNotFoundError(f"File not found: {weights_path}")

    device = torch.device('cpu')
    pretrained_weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(pretrained_weights, strict=False)
    model.eval()
    logging.info("Successfully loaded weights and set model to evaluation mode.")
    return model

def save_model_to_bento(model: MTRCNN_pad_half, model_name: str, modules_to_bundle: List[ModuleType]) -> None:
    """
    Saves the PyTorch model to the BentoML store, bundling external code dependencies.
    """
    logging.info(f"Saving model '{model_name}' to BentoML store.")

    bentoml.pytorch.save_model(
        name=model_name,
        model=model,
        signatures={
            "__call__": {"batchable": True, "batch_dim": 0}
        },
        labels={
            "owner": "ml-team",
            "stage": "production_candidate"
        },
        external_modules=modules_to_bundle
    )
    logging.info(
        f"Successfully saved model '{model_name}' with external modules: {[m.__name__ for m in modules_to_bundle]}."
    )

def main():
    """
    Main function to orchestrate the model preparation and saving process.
    """
    norm_file = os.path.join(BASE_PATH, 'med_normalization.pickle')
    weights_file = os.path.join(BASE_PATH, 'event_detector.pth')

    norm_data = load_normalization_data(norm_file)
    model = create_and_prepare_model(norm_data, weights_file)
    
    # Pass the imported 'framework' module in a list to be bundled.
    save_model_to_bento(model, MODEL_NAME, modules_to_bundle=[framework])


if __name__ == "__main__":
    main()