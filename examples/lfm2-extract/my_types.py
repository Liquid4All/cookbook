from enum import Enum
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field

class Category(str, Enum):
    MEDICAL_CONDITION = "MEDICAL_CONDITION"
    MEDICATION = "MEDICATION"
    MEASUREMENT = "MEASUREMENT"

class Dosage(BaseModel):
    text: str = Field(..., description="Dosage information including amount and frequency")

class Value(BaseModel):
    text: str = Field(..., description="The measurement value with units")

# Define specific models to handle the conditional requirements
class MedicationEntity(BaseModel):
    category: Literal[Category.MEDICATION]
    text: str = Field(..., description="The name of the medication")
    dosage: Dosage = Field(..., description="Dosage is required for medications")

class MeasurementEntity(BaseModel):
    category: Literal[Category.MEASUREMENT]
    text: str = Field(..., description="The name of the measurement")
    value: Value = Field(..., description="Value is required for measurements")

class ConditionEntity(BaseModel):
    category: Literal[Category.MEDICAL_CONDITION]
    text: str = Field(..., description="The name of the medical condition")

# Create the Union type for the items
MedicalEntity = Union[MedicationEntity, MeasurementEntity, ConditionEntity]

class ExtractionResult(BaseModel):
    """The root container for the array of extracted entities"""
    entities: List[MedicalEntity]