"""Test script for real-time audio transcription with actual speech timing."""

import argparse
import os
import sys

from .config import Config
from .model_downloader import ModelDownloader
from .model_wrapper import LFM2AudioWrapper

# from .auto_download import download_model_files_and_llama_cpp


def main(
    audio_file: str,
    play_audio: bool = False,
    clean_text: bool = False,
    log_partial_transcripts: str = None,
    typewriter_effect: bool = False,
    typewriter_speed: float = None,
    quantization: str = None,
):
    """Test real-time transcription functionality."""
    config = Config()

    # Override quantization if provided
    if quantization is not None:
        config.quantization = quantization

    # Ensure llama.cpp builds are available
    try:
        model_downloader = ModelDownloader(
            target_dir=config.base_dir, quantization=config.quantization
        )
        model_downloader.download()
    except Exception as e:
        print(f"⚠️  Warning: Failed to auto-download llama.cpp builds: {e}")
        sys.exit(1)

    print("🚀 Testing Real-Time Audio Transcription")
    if clean_text:
        print("🧹 Text cleaning enabled")
    print("=" * 50)

    # Override typewriter settings if provided
    if typewriter_speed is not None:
        config.typewriter_speed = typewriter_speed

    model = LFM2AudioWrapper(model_downloader, config)

    # Validate audio file exists
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        print("💡 Make sure the audio file exists at the specified path")
        return

    try:
        # Process with real-time timing and optional features
        transcription = model.transcribe_with_real_timing(
            audio_file_path=audio_file,
            chunk_duration=2.0,  # 2 second chunks
            overlap=0.5,  # 0.5 second overlap
            play_audio=play_audio,
            clean_text=clean_text,
            log_partial_transcripts=log_partial_transcripts,
            typewriter_effect=typewriter_effect,
        )

        print("\n🎯 Final Result:")
        print(transcription)

    except FileNotFoundError as e:
        print(f"❌ Audio file not found: {e}")
        print("💡 Make sure the audio file exists at the specified path")
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        raise e


def cli():
    """CLI entry point for the transcribe command."""
    parser = argparse.ArgumentParser(description="Real-time audio transcription")
    parser.add_argument(
        "--audio", required=True, help="Path to the audio file to transcribe"
    )
    parser.add_argument(
        "--play-audio",
        action="store_true",
        help="Play audio in background during transcription",
    )
    parser.add_argument(
        "--clean-text",
        action="store_true",
        help="Clean transcription with language model",
    )
    parser.add_argument(
        "--log-partial-transcripts",
        help="CSV file path in datasets/ to log incremental chunk transcriptions",
    )
    parser.add_argument(
        "--typewriter",
        action="store_true",
        default=True,
        help="Enable typewriter effect for character-by-character display (default: enabled)",
    )
    parser.add_argument(
        "--typewriter-speed",
        type=float,
        default=0.01,
        help="Speed of typewriter effect in seconds per character (default: 0.01)",
    )
    parser.add_argument(
        "--quantization",
        choices=["Q8_0", "F16"],
        default=None,
        help="Quantization variant of the model to use (default: Q8_0)",
    )
    args = parser.parse_args()

    main(
        args.audio,
        args.play_audio,
        args.clean_text,
        args.log_partial_transcripts,
        args.typewriter,
        args.typewriter_speed,
        args.quantization,
    )


if __name__ == "__main__":
    cli()
