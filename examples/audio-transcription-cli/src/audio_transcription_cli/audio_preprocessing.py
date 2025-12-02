"""Audio preprocessing module for LFM2 model compatibility."""

import tempfile
from collections.abc import Iterator

import numpy as np
import soundfile as sf


def save_raw_audio_as_wav(audio_data: np.ndarray, sample_rate: int = 48000) -> str:
    """
    Save raw audio data as WAV file for model processing.

    Args:
        audio_data: Raw audio data from microphone
        sample_rate: Sample rate of the audio data

    Returns:
        Path to temporary WAV file
    """
    # Create temporary WAV file
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    import os

    os.close(fd)

    # Save raw audio directly as WAV
    sf.write(temp_path, audio_data, sample_rate, subtype="PCM_16")

    return temp_path


class AudioChunker:
    """Handles chunking of audio files for real-time processing."""

    def __init__(self, chunk_duration: float = 2.0, overlap: float = 0.5):
        """
        Initialize audio chunker.

        Args:
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
        """
        self.chunk_duration = chunk_duration
        self.overlap = overlap

    def get_audio_info(self, audio_file_path: str) -> tuple[float, int, int]:
        """
        Get audio file information.

        Args:
            audio_file_path: Path to audio file

        Returns:
            Tuple of (duration_seconds, sample_rate, total_frames)
        """
        info = sf.info(audio_file_path)
        return info.duration, info.samplerate, info.frames

    def create_chunks(self, audio_file_path: str) -> Iterator[tuple[str, float, float]]:
        """
        Create audio chunks from file with timing information.

        Args:
            audio_file_path: Path to audio file

        Yields:
            Tuple of (chunk_file_path, start_time, end_time)
        """
        # Get audio info
        total_duration, sample_rate, total_frames = self.get_audio_info(audio_file_path)

        print(f"📊 Audio file: {total_duration:.1f}s, {sample_rate}Hz")

        # Calculate chunk parameters
        chunk_frames = int(self.chunk_duration * sample_rate)
        overlap_frames = int(self.overlap * sample_rate)
        step_frames = chunk_frames - overlap_frames

        current_frame = 0
        chunk_index = 0

        while current_frame < total_frames:
            # Calculate chunk boundaries
            start_frame = current_frame
            end_frame = min(current_frame + chunk_frames, total_frames)

            # Calculate timing
            start_time = start_frame / sample_rate
            end_time = end_frame / sample_rate

            # Read audio chunk
            audio_chunk, _ = sf.read(
                audio_file_path, start=start_frame, frames=end_frame - start_frame
            )

            # Save chunk as temporary file
            chunk_path = self._save_chunk(audio_chunk, sample_rate, chunk_index)

            yield chunk_path, start_time, end_time

            # Move to next chunk
            current_frame += step_frames
            chunk_index += 1

            # Break if we've reached the end
            if end_frame >= total_frames:
                break

    def _save_chunk(
        self, audio_data: np.ndarray, sample_rate: int, chunk_index: int
    ) -> str:
        """
        Save audio chunk as temporary WAV file.

        Args:
            audio_data: Audio data for this chunk
            sample_rate: Sample rate
            chunk_index: Index of this chunk

        Returns:
            Path to temporary chunk file
        """
        fd, temp_path = tempfile.mkstemp(
            suffix=f"_chunk_{chunk_index}.wav", prefix="audio_chunk_"
        )
        import os

        os.close(fd)

        # Save chunk
        sf.write(temp_path, audio_data, sample_rate, subtype="PCM_16")

        return temp_path
