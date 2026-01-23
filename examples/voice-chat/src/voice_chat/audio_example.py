import time
import numpy as np

from .audio_recorder import AudioRecorder
from .audio_preprocessor import AudioPreprocessor

# Example usage: Recording and preprocessing together with auto-stop
def record(silence_duration: float = 2.0, silence_threshold: float = 0.01):
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
            return

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
        recorder.save_to_file("processed_audio.wav", processed_int16)

        print(
            f"\nOriginal audio length: {len(audio_data)} samples ({len(audio_data)/16000:.2f}s)"
        )
        print(
            f"Processed audio length: {len(processed_audio)} samples ({len(processed_audio)/16000:.2f}s)"
        )

    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user!")
        audio_data = recorder.stop_recording()
        if len(audio_data) > 0:
            recorder.save_to_file("interrupted_audio.wav", audio_data)
    finally:
        recorder.cleanup()




if __name__ == "__main__":
    # You can customize these parameters
    record(
        silence_duration=20.0,  # Stop after 2 seconds of silence
        silence_threshold=0.01,  # Energy threshold for silence detection
    )
