
from pydantic import BaseModel


class SpeciesDetectionResults(BaseModel):

    most_common_species: str
    species: list[str]
    