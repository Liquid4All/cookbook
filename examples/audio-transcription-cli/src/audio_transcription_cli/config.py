"""Configuration management for audio settings and model paths."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration settings for audio processing and model paths."""

    # Directoy where model files and llama.cpp will be downloaded to
    base_dir: Path = Field(
        default=Path(os.getcwd()) / "LFM2.5-Audio-1.5B-GGUF",
        description="Base directory containing model files",
    )

    # Quantization variant of the model files. The LFM2.5-Audio-1.5B-GGUF repo
    # ships "Q4_0" (~1.1GB), "Q8_0" (default, ~1.8GB) and "F16" (~3.3GB),
    # sizes being the total across the four GGUF files each variant needs.
    quantization: str = Field(
        default="Q8_0", description="Quantization variant of the model files"
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
    def text_cleaner_model_path(self) -> Path:
        """Get full path to text cleaning model file."""
        return self.base_dir / self.text_cleaner_model_filename
