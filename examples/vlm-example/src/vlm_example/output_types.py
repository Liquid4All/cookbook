from typing import Literal

from pydantic import BaseModel


class DefectDetectionOutput(BaseModel):
    answer: Literal["Yes", "No"]
