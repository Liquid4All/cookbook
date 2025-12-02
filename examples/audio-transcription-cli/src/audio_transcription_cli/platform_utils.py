"""Platform detection utilities."""

import platform


def get_platform_info():
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
