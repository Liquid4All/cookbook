import threading
import time
from typing import Callable, Optional
from pathlib import Path
import wave

import numpy as np
import pyaudio

from .audio_preprocessor import AudioPreprocessor

class AudioRecorder:
    """
    Records audio from microphone with real-time streaming capabilities.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        format: int = pyaudio.paInt16,
    ):
        """
        Initialize the audio recorder.

        Args:
            sample_rate: Sampling rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1 for mono)
            chunk_size: Number of frames per buffer (default: 1024)
            format: Audio format (default: 16-bit int)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = format

        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.frames = []
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self.callback: Optional[Callable] = None

        # Silence detection parameters
        self.silence_threshold = 0.01  # Energy threshold for silence
        self.silence_duration = 0.0  # Duration of current silence
        self.max_silence_duration = None  # Max silence before stopping
        self.last_chunk_time = None
        self.has_detected_sound = False  # Only start silence timer after sound is detected

    def start_recording(
        self,
        callback: Optional[Callable] = None,
        silence_duration: Optional[float] = None,
        silence_threshold: float = 0.01,
        show_audio_bar: bool = False,
    ):
        """
        Start recording audio from the microphone.

        Args:
            callback: Optional function to call with each audio chunk
            silence_duration: Stop recording after this many seconds of silence (None = manual stop)
            silence_threshold: Energy threshold below which audio is considered silent
            show_audio_bar: Show a rich audio visualization bar (default: False)
        """
        if self.is_recording:
            print("Already recording!")
            return

        # Set up callback - use rich audio bar if requested, otherwise use provided callback
        if show_audio_bar:
            self.callback = self._rich_audio_bar_callback
        else:
            self.callback = callback
        self.frames = []
        self.is_recording = True
        self.silence_duration = 0.0
        self.max_silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.last_chunk_time = time.time()
        self.has_detected_sound = False

        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            print("Check microphone permissions and audio device availability")
            self.is_recording = False
            return

        self.recording_thread = threading.Thread(target=self._record_loop)
        self.recording_thread.start()
        
        if silence_duration:
            print(
                f"Recording started at {self.sample_rate}Hz... (will auto-stop after {silence_duration}s of silence)"
            )
        else:
            print(f"Recording started at {self.sample_rate}Hz...")

    def _is_silent(self, audio_chunk: np.ndarray) -> bool:
        """
        Determine if an audio chunk is silent based on energy threshold.

        Args:
            audio_chunk: Audio data as numpy array

        Returns:
            True if chunk is silent, False otherwise
        """
        # Normalize to [-1, 1] range
        if audio_chunk.dtype == np.int16:
            normalized = audio_chunk.astype(np.float32) / 32768.0
        else:
            normalized = audio_chunk

        # Calculate RMS energy
        energy = np.sqrt(np.mean(normalized**2))

        return energy < self.silence_threshold

    def _rich_audio_bar_callback(self, chunk, is_silent):
        """
        Rich audio visualization callback with colorful progress bar.
        """
        try:
            from rich.console import Console
            from rich.text import Text
            import sys
            
            # Calculate energy level
            normalized = chunk.astype(np.float32) / 32768.0
            energy = np.sqrt(np.mean(normalized**2))
            
            console = Console(file=sys.stdout, force_terminal=True)
            
            # Amplify the visualization scale for better visibility
            energy_percent = min(energy * 500, 100)  # Scale to 0-100%
            
            # Create colored status indicator (fixed width to prevent shifting)
            if is_silent:
                status = Text("🔇 SILENT  ", style="red bold")
            else:
                status = Text("🎤 SPEAKING", style="green bold")
            
            # Create visual bar using rich characters
            bar_width = 30
            filled_width = int((energy_percent / 100) * bar_width)
            
            if energy_percent > 80:
                bar_style = "red on red"
            elif energy_percent > 50:
                bar_style = "yellow on yellow"
            elif energy_percent > 20:
                bar_style = "green on green"
            else:
                bar_style = "blue on blue"
            
            # Create the bar visualization
            filled_bar = "█" * filled_width
            empty_bar = "░" * (bar_width - filled_width)
            
            bar_text = Text(filled_bar, style=bar_style) + Text(empty_bar, style="dim white")
            
            # Print the complete line with proper overwrite
            output = Text.assemble(
                status, " ",
                Text("Audio: "), bar_text,
                Text(f" {energy_percent:5.1f}% ", style="cyan"),
                Text(f"({energy:.6f})", style="dim")
            )
            
            # Clear line and print new content
            console.print("\r" + " " * 80, end="\r")  # Clear the line first
            console.print(output, end="", highlight=False)
            
        except ImportError:
            # Fallback to simple text if rich is not available
            energy = np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0)**2))
            energy_percent = min(energy * 500, 100)
            status = "SILENT" if is_silent else "SPEAKING"
            bar = "=" * int(energy_percent / 3)  # Simple text bar
            print(f"\r{status} Audio: {bar:<30} {energy_percent:5.1f}%", end="", flush=True)

    def _record_loop(self):
        """Internal loop for recording audio chunks."""
        while self.is_recording:
            try:
                current_time = time.time()
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.frames.append(data)

                # Convert to numpy array for analysis
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                # Check for silence if auto-stop is enabled
                if self.max_silence_duration is not None:
                    is_silent = self._is_silent(audio_chunk)

                    if not is_silent:
                        # Sound detected - start tracking silence from now on
                        self.has_detected_sound = True
                        self.silence_duration = 0.0
                    elif self.has_detected_sound:
                        # Only count silence after we've heard sound
                        elapsed = current_time - self.last_chunk_time
                        self.silence_duration += elapsed

                        if self.silence_duration >= self.max_silence_duration:
                            print(
                                f"\nSilence detected for {self.silence_duration:.1f}s. Stopping recording..."
                            )
                            self.is_recording = False
                            break

                self.last_chunk_time = current_time

                # Call callback with the audio chunk if provided
                if self.callback:
                    callback_is_silent = is_silent if self.max_silence_duration else self._is_silent(audio_chunk)
                    self.callback(audio_chunk, callback_is_silent)

            except Exception as e:
                print(f"Error during recording: {e}")
                break

    def stop_recording(self) -> np.ndarray:
        """
        Stop recording and return the recorded audio.

        Returns:
            numpy array of the recorded audio
        """
        was_recording = self.is_recording or len(self.frames) > 0
        
        if not was_recording:
            print("Not currently recording!")
            return np.array([])

        self.is_recording = False

        if self.recording_thread:
            self.recording_thread.join()

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if not was_recording:
            print("Recording stopped.")

        # Convert frames to numpy array
        audio_data = b"".join(self.frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        return audio_array

    def save_to_file(self, filename: str, audio_data: np.ndarray):
        """
        Save recorded audio to a WAV file.

        Args:
            filename: Output filename
            audio_data: Audio data as numpy array
        """
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

        print(f"Audio saved to {filename}")

    def cleanup(self):
        """Clean up PyAudio resources."""
        if self.is_recording:
            self.stop_recording()
        self.audio.terminate()


# Example usage: Recording and preprocessing together with auto-stop
def record(silence_duration: float = 2.0, silence_threshold: float = 0.01) -> Optional[Path]:
    """
    Record audio and automatically stop when user is silent for specified duration.

    Args:
        silence_duration: Seconds of silence before auto-stopping (default: 2.0)
        silence_threshold: Energy threshold for silence detection (default: 0.01)
    """
    # Initialize recorder and preprocessor
    recorder = AudioRecorder(sample_rate=16000, channels=1)
    preprocessor = AudioPreprocessor(sample_rate=16000)

    # Real-time preprocessing callback
    def process_chunk(chunk, is_silent):
        # Normalize chunk in real-time
        normalized = preprocessor.normalize(chunk)
        energy = np.sqrt(np.mean(normalized**2))

        # Visual feedback
        silence_indicator = "[SILENT]" if is_silent else "[SPEAKING]"
        bar_length = int(energy * 50)
        bar = "=" * bar_length
        print(
            f"\r{silence_indicator} Energy: {bar:<50} {energy:.4f}", end="", flush=True
        )

    try:
        print(f"Silence threshold set to: {silence_threshold}")
        print(f"Will auto-stop after {silence_duration} seconds of silence")
        print(f"Recording will save as: processed_audio.wav")
        print("Start speaking...\n")

        # Start recording with auto-stop on silence
        recorder.start_recording(
            callback=process_chunk,
            silence_duration=silence_duration,
            silence_threshold=silence_threshold,
        )

        # Wait for recording to complete (auto-stop or manual interrupt)
        while recorder.is_recording:
            time.sleep(0.1)

        # Get recorded audio
        audio_data = recorder.stop_recording()

        if len(audio_data) == 0:
            print("\nNo audio recorded!")
            return None

        # Full preprocessing
        print("\n\nApplying full preprocessing pipeline...")
        processed_audio = preprocessor.preprocess(
            audio_data,
            normalize=True,
            remove_dc=True,
            bandpass=True,
            denoise=True,
            trim=True,
        )

        # Save both raw and processed audio
        recorder.save_to_file("raw_audio.wav", audio_data)

        # Convert back to int16 for saving
        processed_int16 = (processed_audio * 32767).astype(np.int16)
        output_file = "processed_audio.wav"
        recorder.save_to_file(output_file, processed_int16)

        print(
            f"\nOriginal audio length: {len(audio_data)} samples ({len(audio_data)/16000:.2f}s)"
        )
        print(
            f"Processed audio length: {len(processed_audio)} samples ({len(processed_audio)/16000:.2f}s)"
        )
        
        return Path(output_file).resolve()

    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user!")
        audio_data = recorder.stop_recording()
        if len(audio_data) > 0:
            output_file = "interrupted_audio.wav"
            recorder.save_to_file(output_file, audio_data)
            return Path(output_file).resolve()
        return None
    finally:
        recorder.cleanup()