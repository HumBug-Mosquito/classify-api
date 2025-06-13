
import os
import sys

import torch
from torch import nn

from src.lib import utils
from src.lib.event_detection_result import (
                                            EventDetectionResults,
                                            cast_to_audio_segments)
from src.lib.model import AudioInputRequirements, SupportedModel

class Med1(SupportedModel):
    """
    Class to represent the MED-1 model.
    """
    model: nn.Module
    device: torch.device

    def __init__(self):
        id = "med-1"
        name = "MED-1"
        description = "Mosquito Event detection model developed for the Humbug 2 Project"
        species = "mosquito"
        requirements = AudioInputRequirements(min_duration=2, sample_rate=8000)
        type = "event_detection"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.mps.is_available()  else 'cpu')

        super().__init__(
            id=id,
            name=name,
            description=description,
            species=species,
            requirements=requirements,
            type=type
        )

    def load(self):
        """
        Load the model.
        """
        # Load the model here

        parent_directory_of_file = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(parent_directory_of_file)

        from framework.models import MTRCNN_pad_half

        normalization_file = os.path.join(os.getcwd(), f'{parent_directory_of_file}/med_normalization.pickle')
        model = MTRCNN_pad_half(
            class_num=1,
            dropout=0.2,
            MC_dropout=True,
            batchnormal=True,
            normalization_mel_file=normalization_file
        )

        model_path = os.path.join(os.getcwd(), f'{parent_directory_of_file}/event_detector.pth')
        pretrained_model = torch.load(model_path, map_location=self.device)
        model.load_state_dict(pretrained_model, strict=False)
        self.model = model.to(self.device)

    def predict(self, audio) -> EventDetectionResults:
        """
        Predict mosquito events in the audio
        """

        with torch.no_grad():
            self.model.eval()
            predictions, seconds_taken = utils.time_function(
                self.model, torch.tensor(audio).to(self.device)
            )

        predictions = predictions.numpy()

        segments = cast_to_audio_segments(
            predictions=predictions,
            prediction_interval=1
        )

        return EventDetectionResults(
            raw = predictions,
            species=self.species,
            segments=segments,
            model=self.id,
            time_taken=seconds_taken
        )