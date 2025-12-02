"""Audio playback module for synchronizing audio with real-time transcription."""

import os
import threading
from pathlib import Path


class AudioPlayer:
    """Audio player for background playback during transcription."""

    def __init__(self, audio_file_path: str | Path):
        """
        Initialize audio player.

        Args:
            audio_file_path: Path to audio file to play
        """
        self.audio_file_path = str(audio_file_path)
        self._playback_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_playing = False

        # Validate file exists
        if not os.path.exists(self.audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {self.audio_file_path}")

    def start_playback(self) -> None:
        """Start audio playback in background thread."""
        if self._is_playing:
            return

        self._stop_event.clear()
        self._playback_thread = threading.Thread(
            target=self._playback_worker, daemon=True
        )
        self._playback_thread.start()

    def stop_playback(self) -> None:
        """Stop audio playback and cleanup."""
        if not self._is_playing:
            return

        self._stop_event.set()

        # Wait for thread to finish
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)

        self._is_playing = False

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing

    def _playback_worker(self) -> None:
        """Worker thread for audio playback."""
        try:
            import time

            import pygame

            # Initialize pygame mixer
            pygame.mixer.init()

            self._is_playing = True

            # Load and play audio
            pygame.mixer.music.load(self.audio_file_path)
            pygame.mixer.music.play()

            # Wait for completion or stop signal
            while pygame.mixer.music.get_busy() and not self._stop_event.is_set():
                time.sleep(0.1)

        except Exception as e:
            if not self._stop_event.is_set():
                print(f"⚠️ Audio playback error: {e}")
        finally:
            self._is_playing = False
            try:
                import pygame

                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except:
                pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.stop_playback()


def create_audio_player(audio_file_path: str | Path) -> AudioPlayer | None:
    """
    Create audio player instance.

    Args:
        audio_file_path: Path to audio file

    Returns:
        AudioPlayer instance or None if creation fails
    """
    try:
        return AudioPlayer(audio_file_path)
    except Exception as e:
        print(f"⚠️ Audio playback not available: {e}")
        return None
