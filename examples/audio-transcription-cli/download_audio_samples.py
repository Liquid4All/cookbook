#!/usr/bin/env python3
"""Download audio samples for the transcription CLI demo."""

import os
import urllib.request
from pathlib import Path


def download_file(url: str, filename: str, target_dir: str) -> bool:
    """
    Download a file from URL to target directory.

    Args:
        url: URL to download from
        filename: Name to save the file as
        target_dir: Directory to save the file in

    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        # Create target directory if it doesn't exist
        Path(target_dir).mkdir(parents=True, exist_ok=True)

        target_path = os.path.join(target_dir, filename)

        # Check if file already exists
        if os.path.exists(target_path):
            print(f"✅ Audio sample already exists: {target_path}")
            return True

        print(f"🔄 Downloading {filename} from {url}")

        # Add User-Agent header to avoid 403 Forbidden errors
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            },
        )

        with urllib.request.urlopen(request) as response:
            with open(target_path, "wb") as f:
                f.write(response.read())

        print(f"✅ Successfully downloaded: {target_path}")
        return True

    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False


def main():
    """Download audio samples for demo purposes."""

    # Configuration
    AUDIO_SAMPLES_DIR = "audio-samples"

    # Audio samples to download
    audio_samples = [
        {
            "url": "https://www.americanrhetoric.com/mp3clips/barackobama/barackobamafederalplaza.mp3",
            "filename": "barackobamafederalplaza.mp3",
        }
    ]

    print("🎵 Downloading audio samples for transcription demo")
    print("=" * 50)

    success_count = 0
    total_count = len(audio_samples)

    for sample in audio_samples:
        success = download_file(
            url=sample["url"], filename=sample["filename"], target_dir=AUDIO_SAMPLES_DIR
        )
        if success:
            success_count += 1

    print(
        f"\n🎯 Download complete: {success_count}/{total_count} files downloaded successfully"
    )

    if success_count == total_count:
        print(f"🚀 All audio samples ready in '{AUDIO_SAMPLES_DIR}/' directory")
        print("💡 You can now run transcription with:")
        print(
            f"   uv run transcribe --audio './{AUDIO_SAMPLES_DIR}/barackobamafederalplaza.mp3' --play-audio"
        )
    else:
        print(
            "⚠️  Some downloads failed. Please check your internet connection and try again."
        )


if __name__ == "__main__":
    main()
