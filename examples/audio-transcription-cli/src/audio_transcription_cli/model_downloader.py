"""Automatic download functionality for llama.cpp builds."""

import os
import stat
import subprocess
import shutil
import sys
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download

from .platform_utils import get_platform_info


class ModelDownloader:
    """Downloader for Liquid AI Audio models and llama.cpp builds necessary to use them"""
    REPO_URL = "https://huggingface.co/LiquidAI/LFM2-Audio-1.5B-GGUF"
    SUPPORTED_PLATFORMS = [
        'android-arm64',
        'macos-arm64', 
        'ubuntu-arm64',
        'ubuntu-x64'
    ]

    def __init__(
        self,
        target_dir: str,
        quantization: str = "Q8_0"
    ):
        self.target_dir = target_dir
        self.quantization = quantization

        self.platform = self._get_platform_info()
        if self.platform not in self.SUPPORTED_PLATFORMS:
            raise ValueError(f"Unsupported platform: {self.platform}")

        self.model_filename = f"LFM2-Audio-1.5B-{quantization}.gguf"
        self.mmproj_filename = f"mmproj-audioencoder-LFM2-Audio-1.5B-{quantization}.gguf"
        self.audiodecoder_filename = f"audiodecoder-LFM2-Audio-1.5B-{quantization}.gguf"
        self.llama_binary_name = "llama-lfm2-audio"

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
        if not self._extract_platform_binaries():
            return False
        
        # Step 3: Fix binary permissions
        print("🔧 Step 3: Fixing binary permissions...")
        if not self._fix_binary_permissions():
            print("⚠️  Warning: Could not fix binary permissions, but continuing...")
        
        print("🎉 Download completed successfully!")
        return True
    
    def _validate_existing_download(self) -> bool:
        """Check if the target directory contains a valid download."""
        target_path = Path(self.target_dir)
        
        # Check for model files
        model_files = [
            target_path / self.model_filename,
            target_path / self.mmproj_filename,
            target_path / self.audiodecoder_filename
        ]
        
        for model_file in model_files:
            if not model_file.exists():
                return False
        
        # Check for platform-specific binary
        binary_path = target_path / "runners" / self.platform / "bin" / self.llama_binary_name
        if not binary_path.exists():
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
                local_dir_use_symlinks=False  # Download actual files, not symlinks
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
    
    def _extract_platform_binaries(self) -> bool:
        """Extract the platform-specific llama.cpp binaries from zip file."""
        try:
            zip_filename = f"llama.cpp-{self.platform}.zip"
            zip_path = Path(self.target_dir) / zip_filename
            
            if not zip_path.exists():
                print(f"❌ Platform zip file not found: {zip_path}")
                available_zips = list(Path(self.target_dir).glob("llama.cpp-*.zip"))
                print(f"💡 Available platforms: {[z.stem.replace('llama.cpp-', '') for z in available_zips]}")
                return False
            
            # Extract to runners directory
            runners_dir = Path(self.target_dir) / "runners"
            platform_dir = runners_dir / self.platform
            platform_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📦 Extracting {zip_filename} to {platform_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(platform_dir)
            
            print(f"✅ Extracted platform binaries to {platform_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error extracting platform binaries: {e}")
            return False
    
    def _fix_binary_permissions(self) -> bool:
        """Fix execute permissions for the extracted binaries."""
        return fix_binary_permissions(self.target_dir)

    def _get_platform_info(self):
        
        import platform

        """Detect the current platform and architecture."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Normalize architecture names
        if machine in ['x86_64', 'amd64']:
            arch = 'x64'
        elif machine in ['aarch64', 'arm64']:
            arch = 'arm64'
        elif machine.startswith('arm'):
            arch = 'arm64'  # Assume arm64 for ARM variants
        else:
            arch = machine
        
        # Normalize system names
        if system == 'darwin':
            platform_name = 'macos'
        elif system == 'linux':
            platform_name = 'ubuntu'  # Assume Ubuntu-compatible for Linux
        else:
            platform_name = system
        
        return f"{platform_name}-{arch}"


    
def validate_platform_support(platform_string: str) -> bool:
    """Check if the current platform is supported by available runners."""
    supported_platforms = [
        'android-arm64',
        'macos-arm64', 
        'ubuntu-arm64',
        'ubuntu-x64'
    ]
    
    return platform_string in supported_platforms


def fix_binary_permissions(target_dir: str) -> bool:
    """
    Fix execute permissions for binary files in runners directories.
    
    Args:
        target_dir: Base directory containing the downloaded files
        
    Returns:
        bool: True if permissions were fixed successfully
    """
    try:
        runners_dir = Path(target_dir) / "runners"
        if not runners_dir.exists():
            print("⚠️  No runners directory found, skipping permission fix")
            return True
        
        print("🔧 Fixing binary permissions...")
        
        # Find all binary files in runners subdirectories
        for platform_dir in runners_dir.iterdir():
            if platform_dir.is_dir():
                bin_dir = platform_dir / "bin"
                if bin_dir.exists():
                    for binary_file in bin_dir.iterdir():
                        if binary_file.is_file():
                            # Add execute permissions for owner, group, and others
                            current_permissions = binary_file.stat().st_mode
                            new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                            binary_file.chmod(new_permissions)
                            print(f"✅ Fixed permissions for: {binary_file}")
        
        print("🎯 Binary permissions fixed successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️  Warning: Could not fix binary permissions: {e}")
        print("💡 You may need to manually run: chmod +x ./LFM2-Audio-1.5B-GGUF/runners/*/bin/*")
        return False


def clone_huggingface_repo(repo_url: str, target_dir: str) -> bool:
    """Download the Hugging Face repository with all files including LFS."""
    try:
        # Extract repo_id from URL
        repo_id = repo_url.replace("https://huggingface.co/", "")
        
        print(f"🔄 Downloading llama.cpp builds from: {repo_url}")
        
        # Try using huggingface_hub first (preferred method)
        print("📦 Using huggingface_hub for optimal download...")
        # Download to cache first, then get the cache path
        cache_dir = snapshot_download(
            repo_id=repo_id,
            local_dir_use_symlinks=False  # Download actual files, not symlinks
        )
        
        # Move from cache to target directory to avoid duplication
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.move(cache_dir, target_dir)
        
        print(f"✅ Successfully downloaded builds to {target_dir}")
        return True
            
    except Exception as e:
        print(f"❌ Error downloading repository: {e}")
        print("💡 Try installing huggingface_hub: pip install huggingface_hub")
        return False


def download_model_files_and_llama_cpp(target_dir: str) -> bool:
    """
    Automatically download llama.cpp builds for LFM2-Audio-1.5B model.
    
    Returns:
        bool: True if builds are available (either already existed or successfully downloaded),
              False if download failed.
    """
    
    # Configuration
    REPO_URL = "https://huggingface.co/LiquidAI/LFM2-Audio-1.5B-GGUF"
    
    # Detect current platform first
    current_platform = get_platform_info()
    print(f"🔍 Detected platform: {current_platform}")
    
    # Validate platform support (blocking - exit if not supported)
    if not validate_platform_support(current_platform):
        print(f"❌ ERROR: Your platform ({current_platform}) is not supported.")
        print("   Supported platforms: android-arm64, macos-arm64, ubuntu-arm64, ubuntu-x64")
        print("   Please wait for builds to be released for your platform.")
        sys.exit(1)
    
    print(f"✅ Platform {current_platform} is supported!")


    # HERE

    # Check if target directory already exists
    if os.path.exists(target_dir):
        runners_dir = Path(target_dir) / "runners"
        if runners_dir.exists():
            print(f"✅ llama.cpp builds already available at: {runners_dir}")

            # Fix binary permissions just in case
            fix_binary_permissions(target_dir)

            return True
        else:
            print(f"🧹 Removing incomplete download directory: {target_dir}")
            shutil.rmtree(target_dir)
    
    # Clone the repository
    success = clone_huggingface_repo(REPO_URL, target_dir)

    if success:
                
        runners_dir = Path(target_dir) / "runners"
        if runners_dir.exists():
            print(f"🎯 llama.cpp builds ready! Runners available at: {runners_dir}")

            # Fix binary permissions after download
            fix_binary_permissions(target_dir)

            return True
        else:
            print(f"⚠️  Warning: runners directory not found in {target_dir}")
            print("    The repository structure may have changed.")
            return False
    else:
        print("❌ Failed to download llama.cpp builds.")
        print("💡 You can try running 'uv run download_llama_cpp_builds.py' manually.")
        return False