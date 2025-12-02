"""Model wrapper for llama-lfm2-audio binary integration."""

import os
import subprocess
import time
from pathlib import Path

from .audio_preprocessing import AudioChunker
from .config import Config
from .model_downloader import ModelDownloader

class LFM2AudioWrapper:
    """Wrapper for llama-lfm2-audio binary."""

    def __init__(self, model_downloader: ModelDownloader,  config: Config):
        """
        Initialize the model wrapper.

        Args:
            model_downloader: ModelDownloader object with model paths and settings
        """
        self.model_downloader = model_downloader
        self.config = config

        # # Validate configuration
        # if not self.model_downloader.validate_paths():
        #     raise ValueError("Invalid configuration: missing required files")

    def transcribe_audio_file(self, audio_file_path: str | Path) -> str:
        """
        Transcribe audio file to text using LFM2 model.

        Args:
            audio_file_path: Path to audio file

        Returns:
            Transcribed text

        Raises:
            RuntimeError: If transcription fails
        """
        audio_path = str(audio_file_path)

        # Verify input file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Get command arguments
        # cmd = self.config.get_model_command(audio_path)
        cmd = self.model_downloader.get_model_command(audio_path)

        try:
            # Run the model from current working directory (not base_dir)
            # since paths in cmd are already absolute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=False,  # Get bytes to handle encoding issues properly
                timeout=30,  # 30 second timeout
                check=False,  # Don't raise exception on non-zero return code
            )

            if result.returncode != 0:
                error_msg = f"Model execution failed with code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr}"
                raise RuntimeError(error_msg)

            # Extract transcription from stdout
            transcription = self._parse_output(result.stdout)
            return transcription

        except subprocess.TimeoutExpired:
            raise RuntimeError("Model execution timed out")
        except Exception as e:
            raise RuntimeError(f"Model execution failed: {str(e)}")

    def _parse_output(self, output: bytes) -> str:
        """
        Parse model output to extract transcription.

        Args:
            output: Raw stdout from model as bytes

        Returns:
            Cleaned transcription text
        """
        # Decode bytes to string
        try:
            output_str = output.decode("utf-8", errors="replace")
        except Exception:
            output_str = str(output)

        # The model output may contain various information
        # We need to extract just the transcription
        lines = output_str.strip().split("\n")

        # Look for the actual transcription in the output
        # This may need adjustment based on actual model output format
        transcription_lines = []
        capturing = False

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip model loading messages
            if any(
                keyword in line.lower()
                for keyword in [
                    "loading",
                    "model",
                    "load_gguf",
                    "loaded",
                    "tensors",
                    "gguf",
                    "encoding",
                    "slice",
                ]
            ):
                continue

            # Skip timing/performance info
            if "ms" in line or "tokens" in line or "speed" in line:
                continue

            # This should be the transcription
            transcription_lines.append(line)

        # Join all transcription lines
        transcription = " ".join(transcription_lines).strip()

        # Clean up the transcription
        transcription = self._clean_transcription(transcription)

        return transcription

    def _clean_transcription(self, text: str) -> str:
        """
        Clean up transcription text.

        Args:
            text: Raw transcription text

        Returns:
            Cleaned transcription
        """
        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove common artifacts that might appear in output
        artifacts = [
            "Perform ASR.",
            "[INST]",
            "[/INST]",
            "<s>",
            "</s>",
            "System:",
            "User:",
            "Assistant:",
            "load_gguf:",
            "Loaded",
            "tensors",
            "encoding audio slice...",
        ]

        for artifact in artifacts:
            text = text.replace(artifact, "")

        return text.strip()

    def transcribe_audio_data(self, audio_data, sample_rate: int = 48000) -> str:
        """
        Transcribe audio data (numpy array) to text.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data

        Returns:
            Transcribed text
        """
        # Import here to avoid circular imports
        from .audio_preprocessing import save_raw_audio_as_wav

        # Save audio data to temporary file
        temp_file = save_raw_audio_as_wav(audio_data, sample_rate)

        try:
            # Transcribe the temporary file
            transcription = self.transcribe_audio_file(temp_file)
            return transcription
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except OSError:
                pass  # File might already be deleted

    def test_model(self) -> bool:
        """
        Test if the model is working correctly.

        Returns:
            True if model test passes, False otherwise
        """
        try:
            # Test with the example audio file
            example_path = self.config.base_dir / "examples" / "harvard.wav"

            if not example_path.exists():
                print(f"Example file not found: {example_path}")
                return False

            print("Testing model with example audio...")
            transcription = self.transcribe_audio_file(example_path)

            print(f"Test transcription: {transcription}")

            # Check if we got a reasonable transcription
            if len(transcription) > 5:  # At least some text
                print("✅ Model test passed")
                return True
            else:
                print("❌ Model test failed: empty transcription")
                return False

        except Exception as e:
            print(f"❌ Model test failed: {e}")
            return False

    def transcribe_with_real_timing(
        self,
        audio_file_path: str | Path,
        chunk_duration: float = 2.0,
        overlap: float = 0.5,
        play_audio: bool = False,
        clean_text: bool = False,
        log_partial_transcripts: str | None = None,
        typewriter_effect: bool = False,
    ) -> str:
        """
        Transcribe audio file with real-time processing that respects actual speech timing.

        Args:
            audio_file_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            play_audio: Whether to play audio in background during transcription
            clean_text: Whether to clean transcription with language model
            log_partial_transcripts: CSV file path to log incremental transcriptions
            typewriter_effect: Whether to display text with typewriter effect

        Returns:
            Complete transcription (cleaned if clean_text=True)
        """
        audio_path = str(audio_file_path)

        # Verify input file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Initialize chunker
        chunker = AudioChunker(chunk_duration=chunk_duration, overlap=overlap)

        # Get audio info
        total_duration, _, _ = chunker.get_audio_info(audio_path)

        print(f"🎵 Starting real-time transcription of {audio_path}")
        print(f"📊 Duration: {total_duration:.1f}s | Chunk size: {chunk_duration}s")
        print("📝 Real-time transcription:")
        print("-" * 60)
        print("📝 ", end="", flush=True)  # Start the line

        raw_transcription_parts = []  # Store raw chunks for context
        already_displayed_parts = []  # Track what's shown on console
        chunk_files_to_cleanup = []
        cleaning_line_count = 0  # Track console lines used for cleaning output

        # Initialize text cleaner BEFORE audio to minimize delay
        text_cleaner = None
        if clean_text:
            print("🧹 Loading text cleaner...")
            init_start = time.time()
            try:
                from .text_cleaner import create_text_cleaner

                text_cleaner = create_text_cleaner(self.config)
                if text_cleaner:
                    text_cleaner.load_model()  # Pre-load model for faster processing
                    init_time = time.time() - init_start
                    print(f"✅ Text cleaner ready ({init_time:.1f}s)")
                else:
                    print("⚠️ Text cleaner not available, using raw transcription")
            except Exception as e:
                print(f"⚠️ Text cleaner initialization failed: {e}")
                text_cleaner = None

        # Initialize raw transcript logger if requested
        raw_transcript_logger = None
        if log_partial_transcripts:
            try:
                from .raw_transcript_logger import RawTranscriptLogger

                raw_transcript_logger = RawTranscriptLogger(log_partial_transcripts)
                print("📊 Partial transcript logging enabled")
            except Exception as e:
                print(f"⚠️ Raw transcript logger initialization failed: {e}")
                raw_transcript_logger = None

        # Initialize audio player AFTER models are ready
        audio_player = None
        if play_audio:
            from .audio_playback import create_audio_player

            audio_player = create_audio_player(audio_path)
            if audio_player:
                print("🔊 Starting audio playback...")
                audio_player.start_playback()

        # Start timing after all initialization is complete
        start_time = time.time()

        # Process all chunks in unified loop
        for chunk_path, chunk_start, chunk_end in chunker.create_chunks(audio_path):
            chunk_files_to_cleanup.append(chunk_path)

            # Calculate when this chunk should be processed (real-time simulation)
            expected_time = start_time + chunk_start
            current_time = time.time()

            # Wait if we're processing too fast (maintain real-time pace)
            if current_time < expected_time:
                wait_time = expected_time - current_time
                time.sleep(wait_time)

            # Process chunk
            # breakpoint()
            chunk_transcription = self.transcribe_audio_file(chunk_path)

            # Log incremental transcription if logger is available
            if raw_transcript_logger and chunk_transcription.strip():
                raw_transcript_logger.log_incremental_chunk(chunk_transcription)

            if chunk_transcription.strip():
                # Accumulate raw transcription for context
                raw_transcription_parts.append(chunk_transcription)

                # Clean with context of last 2 chunks + current if text cleaner available
                new_content = ""
                if text_cleaner:
                    raise Exception("Not implemented yet")

                    # Use context window: last 2 chunks + current chunk
                    if len(raw_transcription_parts) >= 3:
                        context_chunks = raw_transcription_parts[-3:]
                    else:
                        context_chunks = raw_transcription_parts

                    context_text = " ".join(context_chunks)
                    cleaned_context = text_cleaner.clean_text(context_text)

                    # Extract only new content
                    new_content = self._extract_new_content(
                        cleaned_context, already_displayed_parts, chunk_transcription
                    )
                else:
                    # No text cleaner - use raw chunk
                    new_content = chunk_transcription

                # Append new content to console if we have any
                if new_content:
                    if typewriter_effect:
                        # Use typewriter effect for displaying new content
                        self._typewriter_display(
                            " " + new_content,
                            speed=self.config.typewriter_speed,
                            respect_words=self.config.typewriter_respect_words,
                        )
                    else:
                        # Standard immediate display
                        print(" " + new_content, end="", flush=True)
                    already_displayed_parts.append(new_content)

        # Stop audio playback
        if audio_player:
            audio_player.stop_playback()

        # Clean up chunk files
        for chunk_file in chunk_files_to_cleanup:
            os.unlink(chunk_file)

        # Get final transcription from displayed parts or raw parts as fallback
        full_transcription = (
            " ".join(already_displayed_parts)
            if already_displayed_parts
            else " ".join(raw_transcription_parts)
        )
        print(f"\n{'-' * 60}")
        print(f"✅ Complete transcription ({time.time() - start_time:.1f}s):")
        print(f"📄 {full_transcription}")

        return full_transcription

    def _typewriter_display(
        self, text: str, speed: float = 0.05, respect_words: bool = True
    ) -> None:
        """
        Display text with typewriter effect - character by character.

        Args:
            text: Text to display
            speed: Seconds per character
            respect_words: Whether to pause slightly at word boundaries
        """
        if not text:
            return

        words = text.split() if respect_words else [text]

        for i, word in enumerate(words):
            # Add space before word (except first)
            if i > 0:
                print(" ", end="", flush=True)
                time.sleep(speed * 0.5)  # Shorter pause for spaces

            # Display each character in the word
            for char in word:
                print(char, end="", flush=True)
                time.sleep(speed)

            # Small pause at word boundaries if enabled
            if respect_words and i < len(words) - 1:
                time.sleep(speed * 1.5)

    def _clear_console_lines(self, line_count: int) -> None:
        """Clear previous console output lines."""
        for _ in range(line_count):
            print("\033[A\033[K", end="")  # Move up one line and clear it

    def _extract_new_content(
        self, cleaned_text: str, displayed_parts: list, raw_chunk: str = ""
    ) -> str:
        """Extract only new content from cleaned text."""
        if not displayed_parts:
            return cleaned_text

        existing_text = " ".join(displayed_parts)
        if cleaned_text.startswith(existing_text.strip()):
            new_content = cleaned_text[len(existing_text) :].strip()
            return new_content
        else:
            # Fallback: use raw chunk if cleaning changed too much
            return raw_chunk
