#!/usr/bin/env python3
"""
File handler for processing invoice images in a watched directory.
"""

import csv
import os
from pathlib import Path
import time
from typing import Any

from loguru import logger
from watchdog.events import FileSystemEventHandler

from invoice_parser.invoice_processor import InvoiceData, InvoiceProcessor

# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
}

CSV_COLUMNS = ["processed_at", "file_path", "utility", "amount", "currency"]


def process_invoice(
    processor: InvoiceProcessor, image_path: str
) -> dict[str, Any] | None:
    """Process a single invoice image and return the extracted data as a dict."""
    try:
        logger.info(f"Processing invoice: {image_path}")
        bill_data_obj: InvoiceData = processor.process(image_path)

        if bill_data_obj is None:
            return None

        bill_data = bill_data_obj.model_dump()
        bill_data.update(
            {
                "file_path": image_path,
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        logger.info(f"Structured data extracted: {bill_data}")
        return bill_data

    except Exception as e:
        logger.error(f"Error processing invoice {image_path}: {e}")
        return None


def append_to_csv(output_file: str, data: dict[str, Any]):
    """Append bill data to a CSV file."""
    try:
        logger.info(f"Appending data to CSV: {output_file}")

        file_exists = os.path.exists(output_file)
        logger.debug(f"File exists: {file_exists}")

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)

            if not file_exists:
                writer.writeheader()

            writer.writerow({col: data.get(col, "") for col in CSV_COLUMNS})

        logger.info("Data appended to CSV successfully.")
    except Exception as e:
        logger.error(f"Error writing to CSV {output_file}: {e}")


class InvoiceFileHandler(FileSystemEventHandler):
    """Handles file system events for new invoice images."""

    def __init__(self, processor: InvoiceProcessor, output_file: str):
        self.processor = processor
        self.output_file = output_file
        self.processed_files: set[str] = set()
        logger.info(f"Output will be saved to: {self.output_file}")

        # Keep for backwards compat with process_existing_files in main.py
        self.image_extensions = IMAGE_EXTENSIONS

    def on_created(self, event):
        """Handle new file creation events."""
        if event.is_directory:
            return

        file_path = str(event.src_path)
        file_ext = Path(file_path).suffix.lower()

        if file_ext in IMAGE_EXTENSIONS and file_path not in self.processed_files:
            logger.info(f"New image detected: {file_path}")
            self._process_and_save(file_path)

    def _process_and_save(self, image_path: str):
        """Process an invoice and append to CSV."""
        self.processed_files.add(image_path)
        bill_data = process_invoice(self.processor, image_path)
        if bill_data is None:
            return

        append_to_csv(self.output_file, bill_data)
        logger.info(
            f"Successfully processed {image_path}: {bill_data['utility']} - {bill_data['amount']}{bill_data['currency']}"
        )

    def process_invoice(self, image_path: str):
        """Process a single invoice image (used by process_existing_files)."""
        self._process_and_save(image_path)
