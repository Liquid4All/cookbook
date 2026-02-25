"""Tests for security.decrypt_file tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from tools.decrypt_file import DecryptFile
from tools.encrypt_file import EncryptFile


@pytest.fixture()
def tool() -> DecryptFile:
    """Create a DecryptFile tool instance."""
    return DecryptFile()


@pytest.fixture()
def encrypt_tool() -> EncryptFile:
    """Create an EncryptFile tool instance for setup."""
    return EncryptFile()


@pytest.fixture()
async def encrypted_file(
    encrypt_tool: EncryptFile, plaintext_file: Path
) -> tuple[Path, bytes]:
    """Encrypt a file and return (encrypted_path, original_content)."""
    original = plaintext_file.read_bytes()
    params = encrypt_tool.get_params_model()(path=str(plaintext_file))
    result = await encrypt_tool.execute(params)
    assert result.data is not None
    return Path(result.data.path), original


async def test_decrypt_roundtrip(
    tool: DecryptFile, encrypted_file: tuple[Path, bytes], tmp_dir: Path
) -> None:
    """Should decrypt an encrypted file back to original content."""
    enc_path, original_content = encrypted_file
    output = str(tmp_dir / "decrypted.txt")

    params = tool.get_params_model()(path=str(enc_path), output_path=output)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert Path(result.data.path).read_bytes() == original_content


async def test_decrypt_default_output_strips_enc(
    tool: DecryptFile, encrypted_file: tuple[Path, bytes]
) -> None:
    """Should default to stripping .enc from the path."""
    enc_path, _original = encrypted_file

    params = tool.get_params_model()(path=str(enc_path))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    # Should strip the .enc extension
    assert not result.data.path.endswith(".enc")
    assert Path(result.data.path).exists()


async def test_decrypt_file_not_found(tool: DecryptFile, tmp_dir: Path) -> None:
    """Should raise error for missing encrypted file."""
    from mcp_base import MCPError

    params = tool.get_params_model()(path=str(tmp_dir / "nonexistent.enc"))
    with pytest.raises(MCPError, match="File not found"):
        await tool.execute(params)


async def test_decrypt_key_not_found(tool: DecryptFile, tmp_dir: Path) -> None:
    """Should raise error when key file is missing."""
    from mcp_base import MCPError

    # Create an encrypted file but delete the key
    enc_file = tmp_dir / "orphan.enc"
    key = Fernet.generate_key()
    fernet = Fernet(key)
    enc_file.write_bytes(fernet.encrypt(b"data"))
    # No key file written

    params = tool.get_params_model()(path=str(enc_file))
    with pytest.raises(MCPError, match="Key file not found"):
        await tool.execute(params)


async def test_decrypt_wrong_key(tool: DecryptFile, tmp_dir: Path) -> None:
    """Should raise error when key does not match."""
    from mcp_base import MCPError

    # Encrypt with one key, store a different key
    key1 = Fernet.generate_key()
    key2 = Fernet.generate_key()
    fernet1 = Fernet(key1)

    enc_file = tmp_dir / "bad_key.enc"
    enc_file.write_bytes(fernet1.encrypt(b"secret data"))

    # Write the wrong key
    key_file = tmp_dir / "bad_key.enc.key"
    key_file.write_bytes(key2)

    params = tool.get_params_model()(path=str(enc_file))
    with pytest.raises(MCPError, match="Decryption failed"):
        await tool.execute(params)


async def test_decrypt_custom_output_path(
    tool: DecryptFile, encrypted_file: tuple[Path, bytes], tmp_dir: Path
) -> None:
    """Should write decrypted output to custom path."""
    enc_path, original_content = encrypted_file
    custom_output = str(tmp_dir / "custom_decrypted.txt")

    params = tool.get_params_model()(path=str(enc_path), output_path=custom_output)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert result.data.path == custom_output
    assert Path(custom_output).read_bytes() == original_content


async def test_decrypt_path_is_directory(tool: DecryptFile, tmp_dir: Path) -> None:
    """Should raise error when path is a directory."""
    from mcp_base import MCPError

    params = tool.get_params_model()(path=str(tmp_dir))
    with pytest.raises(MCPError, match="not a file"):
        await tool.execute(params)


def test_metadata(tool: DecryptFile) -> None:
    """Should have correct tool metadata."""
    assert tool.name == "security.decrypt_file"
    assert tool.confirmation_required is True
    assert tool.undo_supported is False
