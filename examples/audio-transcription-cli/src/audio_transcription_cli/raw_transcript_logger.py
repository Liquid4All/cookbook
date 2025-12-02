"""Raw transcript logger for incremental chunk transcription logging."""

import csv
from pathlib import Path


class RawTranscriptLogger:
    """Logger for incremental concatenation of chunk transcriptions."""

    def __init__(self, csv_path: str):
        """
        Initialize the raw transcript logger.

        Args:
            csv_path: Path to the CSV file where transcriptions will be logged
        """
        self.csv_path = Path(csv_path)
        self.accumulated_text: list[str] = []

    def log_incremental_chunk(self, new_chunk: str) -> None:
        """
        Log a new chunk as part of incremental concatenation.

        Each call adds the new chunk to the accumulated text and logs the
        concatenation of all chunks processed so far.

        Args:
            new_chunk: The new chunk transcription to add
        """
        # Add new chunk to our running list
        self.accumulated_text.append(new_chunk)

        # Create concatenated text of all chunks so far
        full_text = " ".join(self.accumulated_text)

        # Write to CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([full_text])
