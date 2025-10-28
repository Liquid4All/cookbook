"""
Invoice processor module for handling image processing and data extraction.
"""

from loguru import logger
import ollama
from pydantic import BaseModel


class InvoiceText(BaseModel):
    description: str


class InvoiceData(BaseModel):
    utility: str
    amount: float
    currency: str


class InvoiceProcessor:
    """Handles invoice image processing and data extraction."""

    def __init__(self, extractor_model: str, image_process_model: str):
        self.extractor_model = extractor_model
        self.image_process_model = image_process_model

        self._download_model(model=extractor_model)
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
        invoice_text = self.image2text(image_path)
        if not invoice_text:
            logger.warning(f"No text extracted from {image_path}")
            return None

        bill_data = self.text2json(invoice_text)
        if not bill_data:
            logger.warning("No structured data extracted from text.")
            return None

        return bill_data  # InvoiceData.model_validate(bill_data)

    def image2text(self, image_path: str) -> InvoiceText | None:
        """Extract text content from invoice image using vision model."""
        try:
            response = ollama.chat(
                model=self.image_process_model,
                messages=[
                    {
                        "role": "user",
                        "content": "What is the amount to pay in the invoice? Please provide the amount, currency and type of bill in a concise format.",
                        "images": [image_path],
                    }
                ],
                format=InvoiceText.model_json_schema(),
                options={"temperature": 0.0},
            )
            response_content = response["message"]["content"]
            # breakpoint()
            return InvoiceText.model_validate_json(response_content)

        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return None

    def text2json(self, invoice_text: InvoiceText) -> InvoiceData | None:
        """Extract structured bill data (type and amount) from text content."""
        try:
            system_prompt = """Identify and extract the following information. Present as a JSON object.
            
            utility: Type of utility (e.g., electricity, water, gas).
            amount: Amount shown on the bill. Only provide the numeric value.
            currency: Currency of the amount (e.g., USD, EUR).
            """

            response = ollama.chat(
                model=self.extractor_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": invoice_text.description},
                ],
                format=InvoiceData.model_json_schema(),
                options={"temperature": 0.0},
            )

            # # Try to parse JSON response
            # response_content = response['message']['content'].strip()

            # # Try to parse the JSON response
            # try:
            #     json_data = json.loads(response_content)
            #     print(f"✅ Extraction successful!")
            #     print(f"Extracted data: {json_data}")
            #     return json_data
            # except json.JSONDecodeError:
            #     print(f"⚠️ Warning: Response was not valid JSON")
            #     print(f"Raw response: {response_content}")
            #     return None
            invoice_data = InvoiceData.model_validate_json(
                response["message"]["content"]
            )
            return invoice_data

        except Exception as e:
            logger.error(f"Error extracting bill data: {e}")
            return None
