#!/usr/bin/env python3
"""
Invoice extraction tool that processes bill/invoice images using a small Vision Language Model.
Extracts bill type and amount information and appends to CSV files for expense tracking.
"""

from pathlib import Path
import time

import click
from loguru import logger
from watchdog.observers import Observer

from invoice_parser.invoice_file_handler import (
    IMAGE_EXTENSIONS,
    InvoiceFileHandler,
    append_to_csv,
    process_invoice,
)
from invoice_parser.invoice_processor import InvoiceProcessor
from invoice_parser.llama_server import (
    DEFAULT_PORT,
    start_llama_server,
    stop_server,
    wait_for_server,
)
from invoice_parser.table_printer import print_results_table


def process_existing_files(directory: str, handler: InvoiceFileHandler):
    """Process any existing image files in the directory."""
    logger.info(f"Processing existing files in {directory}")

    for file_path in Path(directory).rglob("*"):
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            if file_ext in handler.image_extensions:
                handler.process_invoice(str(file_path))


def collect_image_paths(paths: tuple[str, ...]) -> list[Path]:
    """Collect image file paths from the given files and directories."""
    image_paths: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(file_path)
        elif path.is_file():
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)
            else:
                logger.warning(f"Skipping non-image file: {path}")
        else:
            logger.warning(f"Path does not exist: {path}")
    return image_paths


@click.group()
def cli():
    """Invoice extraction tool using Large Foundation Models.

    Use 'watch' to continuously monitor a directory, or 'process' for one-shot
    processing of specific files or folders.
    """


@cli.command()
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
@click.option(
    "--port",
    default=DEFAULT_PORT,
    type=int,
    help="Port to run the llama-server on",
)
@click.option(
    "--verbose-server",
    is_flag=True,
    help="Show llama-server output",
)
def watch(
    dir: Path,
    image_model: str,
    process_existing: bool,
    port: int,
    verbose_server: bool,
):
    """Watch a directory for new invoice images and process them continuously."""
    logger.info(f"Starting llama-server with model: {image_model}")
    server_process = start_llama_server(image_model, port=port, verbose=verbose_server)

    try:
        wait_for_server(port=port)

        base_url = f"http://127.0.0.1:{port}/v1"
        processor = InvoiceProcessor(image_model, base_url=base_url)
        handler = InvoiceFileHandler(processor, str(dir / "bills.csv"))

        if process_existing:
            process_existing_files(str(dir), handler)

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
    finally:
        stop_server(server_process)


@cli.command()
@click.option(
    "--image-model",
    required=True,
    help="LFM vision model name for image processing and data extraction (e.g., LFM2-VL-3B)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="CSV file to append results to (optional)",
)
@click.option(
    "--port",
    default=DEFAULT_PORT,
    type=int,
    help="Port to run the llama-server on",
)
@click.option(
    "--verbose-server",
    is_flag=True,
    help="Show llama-server output",
)
@click.argument("paths", nargs=-1, required=True)
def process(
    image_model: str,
    output: Path | None,
    port: int,
    verbose_server: bool,
    paths: tuple[str, ...],
):
    """Process specific invoice files or folders and exit.

    Accepts one or more file paths or directory paths. Directories are scanned
    recursively for image files. Results are printed to the console, and
    optionally appended to a CSV file via --output.
    """
    image_paths = collect_image_paths(paths)

    if not image_paths:
        logger.warning("No image files found in the provided paths.")
        return

    logger.info(f"Found {len(image_paths)} image(s) to process")
    logger.info(f"Image processing model: {image_model}")

    logger.info(f"Starting llama-server with model: {image_model}")
    server_process = start_llama_server(image_model, port=port, verbose=verbose_server)

    try:
        wait_for_server(port=port)

        base_url = f"http://127.0.0.1:{port}/v1"
        processor = InvoiceProcessor(image_model, base_url=base_url)

        results = []
        for image_path in image_paths:
            bill_data = process_invoice(processor, str(image_path))
            if bill_data is None:
                continue

            results.append(bill_data)

            if output is not None:
                append_to_csv(str(output), bill_data)

        if results:
            print_results_table(results)

        logger.info("Processing complete.")
    finally:
        stop_server(server_process)


if __name__ == "__main__":
    cli()
