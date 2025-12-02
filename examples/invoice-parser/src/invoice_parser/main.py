#!/usr/bin/env python3
"""
Invoice extraction tool that processes bill/invoice images using Large Foundation Models.
Extracts bill type and amount information and appends to CSV files for expense tracking.
"""

from pathlib import Path
import time

import click
from loguru import logger
from watchdog.observers import Observer

from invoice_parser.invoice_file_handler import InvoiceFileHandler
from invoice_parser.invoice_processor import InvoiceProcessor


def process_existing_files(directory: str, handler: InvoiceFileHandler):
    """Process any existing image files in the directory."""
    logger.info(f"Processing existing files in {directory}")

    for file_path in Path(directory).rglob("*"):
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            if file_ext in handler.image_extensions:
                handler.process_invoice(str(file_path))


@click.command()
@click.option(
    "--dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to watch for invoice images",
)
@click.option(
    "--image-model",
    required=True,
    help="LFM vision model name for image processing and data extraction (e.g., LFM2-VL-3B)",
)
@click.option(
    "--process-existing",
    is_flag=True,
    help="Process existing files in the directory on startup",
)
def main(
    dir: Path,
    image_model: str,
    process_existing: bool,
):
    """Invoice extraction tool using Large Foundation Models.

    This tool watches a directory for new invoice images, processes them using
    an LFM vision model to extract bill type and amount, and saves the data to a CSV file.
    """
    # Initialize processor and handler
    processor = InvoiceProcessor(image_model)
    handler = InvoiceFileHandler(processor, str(dir / 'bills.csv'))

    # Process existing files if requested
    if process_existing:
        process_existing_files(str(dir), handler)

    # Set up file watcher
    observer = Observer()
    observer.schedule(handler, str(dir), recursive=True)

    logger.info("Starting invoice extraction tool...")
    logger.info(f"Watching directory: {dir}")
    logger.info(f"Image processing model: {image_model}")

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping invoice extraction tool...")
        observer.stop()

    observer.join()
    logger.info("Invoice extraction tool stopped.")


if __name__ == "__main__":
    main()
