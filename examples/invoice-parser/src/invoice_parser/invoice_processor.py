"""
Invoice processor module for handling image processing and data extraction.
"""

import base64
import mimetypes

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel


class InvoiceData(BaseModel):
    utility: str
    amount: float
    currency: str


class InvoiceProcessor:
    """Handles invoice image processing and data extraction."""

    def __init__(
        self,
        image_process_model: str,
        base_url: str = "http://127.0.0.1:8080/v1",
    ):
        self.image_process_model = image_process_model
        self.client = OpenAI(base_url=base_url, api_key="not-needed")

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
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = mimetypes.guess_type(image_path)[0] or "image/png"

            response = self.client.chat.completions.create(
                model=self.image_process_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "What is the amount to pay in the invoice? "
                                    "Please provide the amount, currency and type of bill "
                                    "in a concise format. Present as a JSON object. "
                                    "utility: Type of utility (e.g., electricity, water, gas). "
                                    "amount: Amount shown on the bill. Only provide the numeric value. "
                                    "currency: Currency of the amount (e.g., USD, EUR)."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{b64}",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "InvoiceData",
                        "schema": InvoiceData.model_json_schema(),
                    },
                },
            )
            content = response.choices[0].message.content
            if content is None:
                return None
            return InvoiceData.model_validate_json(content)

        except Exception as e:
            logger.error(f"Error extracting data from {image_path}: {e}")
            return None
