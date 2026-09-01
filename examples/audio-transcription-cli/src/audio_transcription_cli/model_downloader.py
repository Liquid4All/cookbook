"""Automatic download functionality for llama.cpp builds."""

import os
import shutil
import stat
import sys
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

from .platform_utils import get_platform_info


class ModelDownloader:
    """Downloader for Liquid AI Audio models and llama.cpp builds necessary to use them"""

    REPO_URL = "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF"
    SUPPORTED_PLATFORMS = ["android-arm64", "macos-arm64", "ubuntu-arm64", "ubuntu-x64"]

    MODEL_NAME = "LFM2.5-Audio-1.5B"
    RUNNER_NAME = "llama-liquid-audio"

    # Each model variant ships as four GGUF files that share a naming scheme:
    # "{prefix}{MODEL_NAME}-{quantization}.gguf". Roles map to filename prefixes.
    FILE_PREFIXES = {
        "model": "",
        "mmproj": "mmproj-",
        "vocoder": "vocoder-",
        "tokenizer": "tokenizer-",
    }

    def __init__(self, target_dir: str, quantization: str = "Q8_0"):
        self.target_dir = target_dir
        self.quantization = quantization

        self.platform = self._get_platform_info()
        if self.platform not in self.SUPPORTED_PLATFORMS:
            raise ValueError(f"Unsupported platform: {self.platform}")

        self.llama_binary_name = f"{self.RUNNER_NAME}-cli"
        self.asr_prompt = "Perform ASR."

        self._warm_up_llama_cpp()

    @property
    def llama_cpp_binary_dir(self) -> Path:
        """Return the path to the runner binaries for the current platform."""
        return (
            Path(self.target_dir)
            / "runners"
            / self.platform
            / f"{self.RUNNER_NAME}-{self.platform}"
        )

    def gguf_path(self, role: str) -> Path:
        """Return the path to the GGUF file for the given role and quantization."""
        prefix = self.FILE_PREFIXES[role]
        return (
            Path(self.target_dir)
            / f"{prefix}{self.MODEL_NAME}-{self.quantization}.gguf"
        )

    @property
    def model_path(self) -> Path:
        """Return the path to the main model file."""
        return self.gguf_path("model")

    @property
    def mmproj_path(self) -> Path:
        """Return the path to the multimodal projector file."""
        return self.gguf_path("mmproj")

    @property
    def vocoder_path(self) -> Path:
        """Return the path to the vocoder file."""
        return self.gguf_path("vocoder")

    @property
    def tokenizer_path(self) -> Path:
        """Return the path to the speaker tokenizer file."""
        return self.gguf_path("tokenizer")
    
    def download(self) -> bool:
        """
        Download the model files and llama.cpp builds necessary to use them

        Steps:
        1. Git clone the repository
        2. Unzip the llama.cpp zip file for the current platform
        3. Fix binary permissions

        Returns:
            bool: True if the download was successful, False otherwise
        """
        print(f"🔍 Detected platform: {self.platform}")
        print(f"🎯 Target directory: {self.target_dir}")

        # Check if target directory already exists with valid content
        if os.path.exists(self.target_dir):
            if self._validate_existing_download():
                print(f"✅ Valid download already exists at: {self.target_dir}")
                
                # Fix binary permissions just in case
                self._make_llama_cpp_binaries_executable()

                return True
            else:
                print(f"🧹 Removing incomplete download directory: {self.target_dir}")
                shutil.rmtree(self.target_dir)

        # Step 1: Clone the repository
        print("📥 Step 1: Cloning repository...")
        if not self._clone_repository():
            return False

        # Step 2: Extract platform-specific binaries
        print("📦 Step 2: Extracting platform-specific binaries...")
        self._extract_llama_cpp_binaries()

        # Step 3: Make downloaded binaries executable
        print("🔧 Step 3: Making binaries executable...")
        self._make_llama_cpp_binaries_executable()

        print("🎉 Download completed successfully!")
        return True

    def get_model_command(self, audio_file_path: str) -> list[str]:
        """
        Get command line arguments for llama-liquid-audio-cli.

        Args:
            audio_file_path: Path to input audio file

        Returns:
            List of command arguments
        """
        return [
            str(self.llama_cpp_binary_dir / self.llama_binary_name),
            "-m",
            str(self.model_path),
            "-mm",
            str(self.mmproj_path),
            "-mv",
            str(self.vocoder_path),
            "--tts-speaker-file",
            str(self.tokenizer_path),
            "-sys",
            self.asr_prompt,
            "--audio",
            audio_file_path,
        ]
    
    def _validate_existing_download(self) -> bool:
        """Check if the target directory contains a valid download."""
        # Check for model files
        model_files = [self.gguf_path(role) for role in self.FILE_PREFIXES]

        for model_file in model_files:
            if not model_file.exists():
                return False

        # Check for platform-specific binary
        if not self.llama_cpp_binary_dir.exists():
            return False

        if not (self.llama_cpp_binary_dir / self.llama_binary_name).exists():   
            return False
        
        return True

    def _clone_repository(self) -> bool:
        """Clone the Hugging Face repository with all files including LFS."""
        try:
            # Extract repo_id from URL
            repo_id = self.REPO_URL.replace("https://huggingface.co/", "")

            print(f"🔄 Downloading llama.cpp builds from: {self.REPO_URL}")

            # Use huggingface_hub for optimal download
            print("📦 Using huggingface_hub for optimal download...")
            # Download to cache first, then get the cache path
            cache_dir = snapshot_download(
                repo_id=repo_id,
                local_dir_use_symlinks=False,  # Download actual files, not symlinks
            )

            # Copy from cache to target directory
            if os.path.exists(self.target_dir):
                shutil.rmtree(self.target_dir)
            # TODO: turn into a move operation so we don't take double space on disk
            shutil.copytree(cache_dir, self.target_dir)

            print(f"✅ Successfully downloaded builds to {self.target_dir}")
            return True

        except Exception as e:
            print(f"❌ Error downloading repository: {e}")
            print("💡 Try installing huggingface_hub: pip install huggingface_hub")
            return False

    def _extract_llama_cpp_binaries(self) -> bool:
        """Extract the platform-specific llama.cpp binaries from zip file."""
        try:
            zip_filename = f"{self.RUNNER_NAME}-{self.platform}.zip"
            zip_path = Path(self.target_dir) / 'runners' / zip_filename

            if not zip_path.exists():
                print(f"❌ Platform zip file not found: {zip_path}")
                available_zips = list(
                    (Path(self.target_dir) / "runners").glob(f"{self.RUNNER_NAME}-*.zip")
                )
                print(
                    f"💡 Available platforms: {[z.stem.replace(f'{self.RUNNER_NAME}-', '') for z in available_zips]}"
                )
                return False

            # Extract to runners directory
            runners_dir = Path(self.target_dir) / "runners"
            platform_dir = runners_dir / self.platform
            platform_dir.mkdir(parents=True, exist_ok=True)

            print(f"📦 Extracting {zip_filename} to {platform_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(platform_dir)

            print(f"✅ Extracted platform binaries to {platform_dir}")
            return True

        except Exception as e:
            print(f"❌ Error extracting platform binaries: {e}")
            return False

    def _make_llama_cpp_binaries_executable(self) -> bool:
        """
        Make the extracted binaries executable.
        
        """
        try:
            for file_path in Path(self.llama_cpp_binary_dir).iterdir():
                if file_path.is_file():
                    current_permissions = file_path.stat().st_mode
                    file_path.chmod(current_permissions | stat.S_IEXEC)
                    print(f"✅ Made executable: {file_path}")
            return True
        except (OSError, PermissionError):
            return False

    def _get_platform_info(self):
        import platform

        """Detect the current platform and architecture."""
        system = platform.system().lower()
        machine = platform.machine().lower()

        # Normalize architecture names
        if machine in ["x86_64", "amd64"]:
            arch = "x64"
        elif machine in ["aarch64", "arm64"]:
            arch = "arm64"
        elif machine.startswith("arm"):
            arch = "arm64"  # Assume arm64 for ARM variants
        else:
            arch = machine

        # Normalize system names
        if system == "darwin":
            platform_name = "macos"
        elif system == "linux":
            platform_name = "ubuntu"  # Assume Ubuntu-compatible for Linux
        else:
            platform_name = system

        return f"{platform_name}-{arch}"

    def _warm_up_llama_cpp(self):
        """Pre-load the model to reduce latency for first transcription call."""
        import subprocess
        import tempfile
        import numpy as np
        import soundfile as sf
        
        try:
            print("🔥 Warming up llama.cpp model...")
            
            # Generate a short silent audio file for warming up
            sample_rate = 16000
            duration = 0.5  # 0.5 seconds of silence
            silence = np.zeros(int(sample_rate * duration), dtype=np.float32)
            
            # Create temporary file for warm-up audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_audio_path = temp_file.name
                
            # Write silent audio to temp file
            sf.write(temp_audio_path, silence, sample_rate)
            
            # Get command for warm-up
            cmd = self.get_model_command(temp_audio_path)
            
            # Run warm-up with short timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=False,
                timeout=10,  # Short timeout for warm-up
                check=False,
            )
            
            # Clean up temp file
            import os
            os.unlink(temp_audio_path)
            
            if result.returncode == 0:
                print("✅ Model warm-up completed successfully")
            else:
                print("⚠️  Model warm-up completed with warnings (this is normal)")
                
        except Exception as e:
            print(f"⚠️  Model warm-up failed: {e} (transcription will still work)")
            # Don't raise - warm-up failure shouldn't prevent initialization