
from pathlib import Path
from typing import Annotated
import bentoml
from bentoml.validators import ContentType
import torch

@bentoml.service(
    traffic={"timeout": 30, "concurrency": 32},
    resources={"cpu": 1, "memory": "200Mi"},
    image=bentoml.images.Image(python_version="3.11")
        .python_packages("bentoml>=1.4.16\n")
        .python_packages("torch>=2.0.1\n")
        .python_packages("torchaudio>=2.0.2\n")
        .python_packages("numpy>=1.23.5\n"),
)
class MSCMTRCNN1Service:
    """
    A service to detect mosquito events using the MTRCNN model.
    This service is specifically designed for the Humbug 2 Project.
    """
    
    def __init__(self):
        self.model = bentoml.pytorch.load_model("med-general-1:w2tzjlsmfkoxncre", device_id='cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.preprocessor = bentoml.depends()

    @bentoml.api()
    def predict(
        self,
        audio_file: Annotated[Path, ContentType("audio/wav")],
    ):
        """
        Predicts mosquito events from audio data.
        """
