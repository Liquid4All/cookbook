"""Configuration management for audio settings and model paths."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration settings for audio processing and model paths."""

    # Directoy where model files and llama.cpp will be downloaded to
    base_dir: Path = Field(
        default=Path(os.getcwd()) / "LFM2-Audio-1.5B-GGUF",
        description="Base directory containing model files",
    )

    # Model files.
    # By default, we use the LFM2-Audio-1.5B-Q8_0.gguf model.
    model_filename: str = Field(
        default="LFM2-Audio-1.5B-Q8_0.gguf", description="Main model file"
    )
    mmproj_filename: str = Field(
        default="mmproj-audioencoder-LFM2-Audio-1.5B-Q8_0.gguf",
        description="Audio encoder projection file",
    )
    audiodecoder_filename: str = Field(
        default="audiodecoder-LFM2-Audio-1.5B-Q8_0.gguf",
        description="Audio decoder file",
    )

    # Binary settings
    runner_platform: str = Field(
        default="macos-arm64", description="Platform for binary runners"
    )
    llama_binary_name: str = Field(
        default="llama-lfm2-audio", description="Name of the llama binary"
    )

    # Audio settings
    sample_rate: int = Field(default=48000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    chunk_size: int = Field(default=1024, description="Audio chunk size for processing")
    recording_duration: float = Field(
        default=3.0, description="Duration in seconds for each audio recording chunk"
    )

    # ASR settings
    asr_prompt: str = Field(
        default="Perform ASR.", description="System prompt for ASR task"
    )

    # Text cleaner model settings
    text_cleaner_model_filename: str = Field(
        default="models/LFM2-700M-Q5_K_M.gguf", description="Text cleaning model file"
    )
    text_cleaning_enabled: bool = Field(
        default=False, description="Enable text cleaning with language model"
    )
    text_cleaning_max_tokens: int = Field(
        default=2048, description="Maximum tokens for text cleaning context"
    )
    text_cleaning_temperature: float = Field(
        default=0.3, description="Temperature for text cleaning inference"
    )
    text_cleaner_system_prompt: str = Field(
        default="""You are an AI assistant that cleans raw text transcripts. Your goal is to take the input text, which may contain repetitions or disfluencies, and produce a grammatically correct, coherent, and natural-sounding cleaned version. Do not add new information or alter the original meaning. The output should be a single, continuous paragraph.""",
        description="System prompt for the text cleaner model",
    )
    text_cleaner_user_prompt: str = Field(
        default="""Clean the following raw text transcript:

{raw_text}""",
        description="User prompt template for the text cleaner model (use {raw_text} placeholder)",
    )

    # Typewriter effect settings
    typewriter_enabled: bool = Field(
        default=False,
        description="Enable typewriter effect for character-by-character display",
    )
    typewriter_speed: float = Field(
        default=0.05, description="Speed of typewriter effect in seconds per character"
    )
    typewriter_respect_words: bool = Field(
        default=True,
        description="Whether to pause at word boundaries during typewriter effect",
    )

    class Config:
        env_prefix = "LIQUID_ASR_"
        case_sensitive = False

    @property
    def model_path(self) -> Path:
        """Get full path to model file."""
        return self.base_dir / self.model_filename

    @property
    def mmproj_path(self) -> Path:
        """Get full path to mmproj file."""
        return self.base_dir / self.mmproj_filename

    @property
    def audiodecoder_path(self) -> Path:
        """Get full path to audiodecoder file."""
        return self.base_dir / self.audiodecoder_filename

    @property
    def llama_binary_path(self) -> Path:
        """Get full path to llama binary."""
        return (
            self.base_dir
            / "runners"
            / self.runner_platform
            / "bin"
            / self.llama_binary_name
        )

    @property
    def text_cleaner_model_path(self) -> Path:
        """Get full path to text cleaning model file."""
        return self.base_dir / self.text_cleaner_model_filename

    # def validate_paths(self) -> bool:
    #     """
    #     Validate that all required files exist.

    #     Returns:
    #         True if all paths are valid, False otherwise
    #     """
    #     required_paths = [
    #         self.model_path,
    #         self.mmproj_path,
    #         self.audiodecoder_path,
    #         self.llama_binary_path,
    #     ]

    #     for path in required_paths:
    #         if not path.exists():
    #             breakpoint()
    #             print(f"Missing required file: {path}")
    #             return False

    #     return True

    def get_model_command(self, audio_file_path: str) -> list[str]:
        """
        Get command line arguments for llama-lfm2-audio.

        Args:
            audio_file_path: Path to input audio file

        Returns:
            List of command arguments
        """
        return [
            str(self.llama_binary_path),
            "-m",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "-mv",
            str(self.audiodecoder_path),
            "-sys",
            self.asr_prompt,
            "--audio",
            audio_file_path,
        ]
