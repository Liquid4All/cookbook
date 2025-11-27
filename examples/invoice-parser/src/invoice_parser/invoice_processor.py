"""
Invoice processor module for handling image processing and data extraction.
"""

from loguru import logger
import ollama
from pydantic import BaseModel


class InvoiceData(BaseModel):
    utility: str
    amount: float
    currency: str


class InvoiceProcessor:
    """Handles invoice image processing and data extraction."""

    def __init__(self, image_process_model: str):
        self.image_process_model = image_process_model

        self._download_model(model=image_process_model)

    def _download_model(self, model: str):
        """Ensure the specified model is downloaded locally."""
        try:
            logger.info(f"Pulling model: {model}")
            ollama.pull(model=model)
            logger.info(f"Model {model} is ready.")
        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")

    def process(self, image_path: str) -> InvoiceData | None:
        """Process an invoice image to extract structured data."""
        invoice_data = self.image2text(image_path)
        if not invoice_data:
            logger.warning(f"No data extracted from {image_path}")
            return None

        return invoice_data

    def image2text(self, image_path: str) -> InvoiceData | None:
        """Extract structured data directly from invoice image using vision model."""
        try:
            response = ollama.chat(
                model=self.image_process_model,
                messages=[
                    {
                        "role": "user",
                        "content": "What is the amount to pay in the invoice? Please provide the amount, currency and type of bill in a concise format. Present as a JSON object. utility: Type of utility (e.g., electricity, water, gas). amount: Amount shown on the bill. Only provide the numeric value. currency: Currency of the amount (e.g., USD, EUR).",
                        "images": [image_path],
                    }
                ],
                format=InvoiceData.model_json_schema(),
                options={"temperature": 0.0},
            )
            response_content = response["message"]["content"]
            return InvoiceData.model_validate_json(response_content)

        except Exception as e:
            logger.error(f"Error extracting data from {image_path}: {e}")
            return None
