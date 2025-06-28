import logging
from pathlib import Path
from typing import Annotated, AsyncGenerator, Generator
import uuid
import bentoml
import bentoml.grpc
import bentoml.grpc.types
from bentoml.validators import ContentType
from circus.config import get_config
from fastapi import FastAPI, WebSocket
import librosa
import numpy as np
import torch
import json

from httpx_ws import WebSocketDisconnect

from common.dto import *
from common.dto_helper import create_dto_from_med_predictions

# Create a stream handler
ch = logging.StreamHandler()

# Set a format for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Get the BentoML logger
bentoml_logger = logging.getLogger("bentoml")

# Add the handler to the BentoML logger
bentoml_logger.addHandler(ch)

# Set the desired logging level (e.g., DEBUG)
bentoml_logger.setLevel(logging.INFO)

app = FastAPI()

@bentoml.service()
class MEDMTRCNN1C2DI_V2_1:
    """
    A service to detect mosquito events using the MTRCNN model.
    This service is specifically designed for the Humbug 2 Project.
    """
    def __init__(self):
        self.model = bentoml.pytorch.load_model("med_mtrcnn_c2di_v2_1:latest", weights_only=False)
        self.model.eval()

    @bentoml.api
    async def predict(self, input: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.model(input).cpu().numpy()  # Move predictions to CPU and convert to numpy array


def get_config_from_message(first_msg: dict) -> tuple[bool, int, int]:

    """
    Extracts configuration from the first message in the WebSocket stream.
    Returns a tuple of (request_id, remove_voices, number_of_samples).
    """

    config = first_msg.get("config", {})

    sample_rate = config.get("sample_rate", 44100)
    remove_voices = config.get("remove_voices", True)
    number_of_samples = config.get("number_of_samples", 10)
    return remove_voices, number_of_samples, sample_rate


@bentoml.service(
    traffic={"timeout": 60, "concurrency": 32}, # Increased timeout for potentially long files/streams
    resources={"cpu": 1, "memory": "200Mi"},
    logging={"access": {
        "enabled": True,
        "request_content_length": True,
        "request_content_type": True,
        "response_content_length": True,
        "response_content_type": True,
        "format": {
            "trace_id": "032x",
            "span_id": "016x"
        }
    }},
)
@bentoml.asgi_app(app)
class MEDMTRCNN1Service:
    """
    A service to detect mosquito events using the MTRCNN model.
    This service now includes both a REST API for file uploads and a WebSocket API for streaming.
    """

    required_sample_rate = 16000
    required_min_duration = 2  # Minimum duration in seconds
    device_id = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_runner = bentoml.depends(MEDMTRCNN1C2DI_V2_1)

    METADATA = {
        "name": "med-mtrcnn-1",
        "description": "MED General 1 model for mosquito event detection.",
        "version": "1.0.0",
        "tags": ["mosquito", "event-detection", "humbug-2", "med-mtrcnn-1"],
        "input_requirements": {
            "sample_rate": required_sample_rate,
            "min_duration": required_min_duration,
        }
    }
     
    def __init__(self):
        with bentoml.importing():
            from common.preprocessor import Preprocessor
            self.preprocessor = Preprocessor(sample_rate=16000, mel_bins=64, vad_threshold=0.2)
        bentoml_logger.info(f"MEDMTRCNN1Service initialized. Device: {self.device_id}")

    async def _process_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        remove_voices: bool,
        number_of_samples: int,
    ) -> tuple[np.ndarray, list[bool] | None]:
        """
        Private helper method containing the core, reusable logic for processing a single audio chunk.
        """
        bentoml_logger.debug(f"Processing audio chunk with shape: {audio_chunk.shape}, dtype: {audio_chunk.dtype}")

        logmel_features = self.preprocessor.extract_features(audio_chunk)
        med_batch = self.preprocessor.prepare_for_event_detection(logmel_features)

        if number_of_samples == 1:
            med_predictions = await self.model_runner.predict(med_batch)
        else:
            predictions_list = []
            for _ in range(number_of_samples):
                predictions_list.append(await self.model_runner.predict(med_batch))
            med_predictions = np.mean(predictions_list, axis=0)

        final_predictions = med_predictions
        vad_mask_for_chunk = None

        if remove_voices:
            _, vad_mask = self.preprocessor.detect_voices(audio_chunk, logmel_features, sample_rate=self.required_sample_rate)
            final_predictions, vad_mask_for_chunk = self.preprocessor.remove_voice_from_med_predictions(med_predictions, vad_mask)
        
        return final_predictions.flatten(), vad_mask_for_chunk

    @bentoml.api(route="/predict")
    async def predict(
        self,
        ctx: bentoml.Context,
        audio_file: Annotated[Path, ContentType("audio/wav")],
        remove_voices: bool = True,
        number_of_samples: int = 3,
    ) -> APIResponse:
        """
        (REST API) Predicts mosquito events from a complete audio file.
        """
        
        request_id = ctx.request.headers.get("X-Request-ID", str(uuid.uuid4()))
        metadata=ResponseMetadata(request_id=request_id)

        try:
            original_sample_rate, duration = librosa.get_samplerate(str(audio_file)), librosa.get_duration(filename=str(audio_file))
        except Exception as e:
            bentoml_logger.error(f"Failed to load audio file: {e}", extra={"request_id": request_id})
            ctx.response.status_code = 400
            return ErrorResponse(metadata=metadata,payload=ErrorPayload(code=400, error="Invalid audio file",message=f"Failed to load audio file: {str(e)}"))

        if duration < self.required_min_duration:
            ctx.response.status_code = 400
            return ErrorResponse(metadata=metadata, payload=ErrorPayload(code=400, error="Audio file too short",message=f"Audio file duration is {duration:.2f} seconds, but at least {self.required_min_duration} seconds is required.",))

        steps: dict[str, str] = {}
        requires_resampling = original_sample_rate != self.required_sample_rate


        audio_generator: Generator[np.ndarray, None, None]
        if requires_resampling:
            try: audio_generator = self.preprocessor.resample(audio_file, target_sr=self.required_sample_rate, chunk_duration_seconds=self.required_min_duration) 
            except Exception as e:
                bentoml_logger.error(f"Failed to resample then load audio file: {e}", extra={"request_id": request_id})
                ctx.response.status_code = 500
                return ErrorResponse(metadata=metadata,payload=ErrorPayload(code=500, error="Resampling error",message=f"Failed to resample audio file: {str(e)}"))
        else: 
            try: audio_generator = self.preprocessor.load_audio(audio_file, chunk_duration_seconds=self.required_min_duration, sample_rate=self.required_sample_rate)
            except Exception as e:
                bentoml_logger.error(f"Failed to load audio file: {e}", extra={"request_id": request_id})
                ctx.response.status_code = 400
                return ErrorResponse(metadata=metadata,payload=ErrorPayload(code=400, error="Invalid audio file",message=f"Failed to load audio bytes sequentially from file: {str(e)}"))

        if requires_resampling: steps["resampling"] = f"Audio resampled from {original_sample_rate}Hz to {self.required_sample_rate}Hz."

        all_predictions = []
        all_vad_masks = []
        for i, audio_chunk in enumerate(audio_generator):
            # Call the reusable helper method
            predictions, vad_mask = await self._process_audio_chunk(audio_chunk, remove_voices, number_of_samples)
            all_predictions.extend(predictions)
            if vad_mask is not None: all_vad_masks.extend(vad_mask)

        if remove_voices: steps["vad"] = "Voices were removed from predictions using VAD."

        return SuccessfulResponse(metadata=metadata, payload=PredictionPayload(
            steps_completed=steps,
            event_detection=create_dto_from_med_predictions(all_predictions, self.METADATA["name"], all_vad_masks),
            species_detection=None,  # No species detection in this service
            message="Prediction completed successfully."
        ))

    @app.websocket("/predict-stream")
    async def predict_websocket(self, websocket: WebSocket):
        """
        Handles audio stream predictions over WebSocket.
        - First message: JSON config.
        - Subsequent messages: Raw float32 audio bytes.
        """
        request_id = str(uuid.uuid4())
        await websocket.accept()
        bentoml_logger.info(f"WebSocket connection accepted for request_id: {request_id}")

        try:
            # 1. Receive and process configuration message
            first_message = await websocket.receive_json()
            remove_voices, number_of_samples, client_sr = get_config_from_message(first_message)
            log_extra = {"request_id": request_id, "client_sample_rate": client_sr}
            bentoml_logger.info(f"Configuration received", extra=log_extra)

            # 2. Process audio stream
            chunk_index = 0
            while True:
                audio_bytes = await websocket.receive_bytes()
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)

                # Resample if necessary
                if client_sr != self.required_sample_rate:
                    audio_chunk = librosa.resample(
                        y=audio_chunk,
                        orig_sr=client_sr,
                        target_sr=self.required_sample_rate
                    )

                # Process the chunk using the existing helper method
                predictions, vad_mask = await self._process_audio_chunk(
                    audio_chunk=audio_chunk,
                    remove_voices=remove_voices,
                    number_of_samples=number_of_samples
                )

                # Create and send response DTO
                event_detection_results = create_dto_from_med_predictions(
                    med_predictions=predictions.tolist(),
                    model_name=self.METADATA.get("name", "undefined"),
                    vad_mask=vad_mask
                )

                response = SuccessfulStreamedResponse(
                    metadata=ResponseMetadata(request_id=request_id),
                    payload= StreamPredictionPayload(
                        chunk_index=chunk_index,
                        steps_completed={"vad": "Voice was removed from the audio clips"} if remove_voices else {},
                        event_detection=event_detection_results,
                        message=f"Chunk {chunk_index} processed successfully."
                    )
                )

                await websocket.send_json(response.model_dump())
                bentoml_logger.debug(f"Sent prediction for chunk {chunk_index}", extra=log_extra)
                chunk_index += 1

        except WebSocketDisconnect:
            bentoml_logger.info(f"Client disconnected gracefully.", extra={"request_id": request_id})
        except json.JSONDecodeError:
            bentoml_logger.warning(f"Failed to decode initial config JSON.", extra={"request_id": request_id})
            await websocket.close(code=1003, reason="Invalid configuration format")
        except Exception as e:
            bentoml_logger.error(f"An unexpected error occurred: {e}", exc_info=True, extra={"request_id": request_id})
            await websocket.close(code=1011, reason="Internal server error")
        finally:
            bentoml_logger.info(f"WebSocket connection closed for request_id: {request_id}")
