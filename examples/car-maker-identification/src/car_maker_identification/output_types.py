from typing import Literal, TypeVar

from pydantic import BaseModel

ModelOutputType = TypeVar("ModelOutputType", bound=BaseModel)


class CarIdentificationOutputType(BaseModel):
    pred_class: Literal[
        "AM",
        "Acura",
        "Aston",
        "Audi",
        "BMW",
        "Bentley",
        "Bugatti",
        "Buick",
        "Cadillac",
        "Chevrolet",
        "Chrysler",
        "Daewoo",
        "Dodge",
        "Eagle",
        "FIAT",
        "Ferrari",
        "Fisker",
        "Ford",
        "GMC",
        "Geo",
        "HUMMER",
        "Honda",
        "Hyundai",
        "Infiniti",
        "Isuzu",
        "Jaguar",
        "Jeep",
        "Lamborghini",
        "Land",
        "Lincoln",
        "MINI",
        "Maybach",
        "Mazda",
        "McLaren",
        "Mercedes-Benz",
        "Mitsubishi",
        "Nissan",
        "Plymouth",
        "Porsche",
        "Ram",
        "Rolls-Royce",
        "Scion",
        "Spyker",
        "Suzuki",
        "Tesla",
        "Toyota",
        "Volkswagen",
        "Volvo",
        "smart",
    ]

    @classmethod
    def from_pred_class(cls, pred_class: str) -> str:
        """Create instance from pred_class and return as JSON string."""
        instance = cls(pred_class=pred_class)
        return instance.model_dump_json()


def get_model_output_schema(dataset_name: str) -> BaseModel:
    if dataset_name == "microsoft/cats_vs_dogs":
        return CatsVsDogsClassificationOutputType
    elif "stanford_cars" in dataset_name:
        return CarIdentificationOutputType
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
