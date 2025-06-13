import numpy as np
from pydantic import BaseModel

from src.lib.event_detection_result import EventDetectionResults


class AudioInputRequirements(BaseModel):
    min_duration: int
    sample_rate: int


class SupportedModel:
    id: str
    name: str
    description: str
    type: str
    species: str
    requirements: AudioInputRequirements

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        species: str,
        type: str,
        requirements: AudioInputRequirements,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.species = species
        self.type = type
        self.requirements = requirements

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "species": self.species,
            "type": self.type,
            "input_requirements": self.requirements.model_dump(),
        }

    def load(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def predict(self, audio: np.ndarray) -> EventDetectionResults:
        raise NotImplementedError("This method should be implemented by subclasses.")