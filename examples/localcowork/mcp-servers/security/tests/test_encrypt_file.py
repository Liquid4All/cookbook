"""Tests for security.encrypt_file tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.encrypt_file import EncryptFile


@pytest.fixture()
def tool() -> EncryptFile:
    """Create an EncryptFile tool instance."""
    return EncryptFile()


async def test_encrypt_creates_enc_file(tool: EncryptFile, plaintext_file: Path) -> None:
    """Should create an encrypted .enc file."""
    params = tool.get_params_model()(path=str(plaintext_file))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    enc_path = Path(result.data.path)
    assert enc_path.exists()
    assert enc_path.name == "secret_document.txt.enc"
    # Encrypted content should differ from plaintext
    assert enc_path.read_bytes() != plaintext_file.read_bytes()


async def test_encrypt_creates_key_file(tool: EncryptFile, plaintext_file: Path) -> None:
    """Should create a .key file alongside the encrypted file."""
    params = tool.get_params_model()(path=str(plaintext_file))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    key_path = Path(result.data.path + ".key")
    assert key_path.exists()
    # Key should be a valid Fernet key (URL-safe base64, 44 chars)
    key_content = key_path.read_bytes()
    assert len(key_content) == 44


async def test_encrypt_returns_key(tool: EncryptFile, plaintext_file: Path) -> None:
    """Should return the Fernet key as identity_public_key."""
    params = tool.get_params_model()(path=str(plaintext_file))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.identity_public_key) == 44

    # Key in result should match key file
    key_path = Path(result.data.path + ".key")
    assert result.data.identity_public_key == key_path.read_text(encoding="utf-8")


async def test_encrypt_custom_output_path(
    tool: EncryptFile, plaintext_file: Path, tmp_dir: Path
) -> None:
    """Should write to a custom output path when specified."""
    output = str(tmp_dir / "custom_encrypted.bin")
    params = tool.get_params_model()(path=str(plaintext_file), output_path=output)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert result.data.path == output
    assert Path(output).exists()
    # Key file should be at custom_encrypted.bin.key
    assert Path(output + ".key").exists()


async def test_encrypt_file_not_found(tool: EncryptFile, tmp_dir: Path) -> None:
    """Should raise error for missing file."""
    from mcp_base import MCPError

    params = tool.get_params_model()(path=str(tmp_dir / "nonexistent.txt"))
    with pytest.raises(MCPError, match="File not found"):
        await tool.execute(params)


async def test_encrypt_path_is_directory(tool: EncryptFile, tmp_dir: Path) -> None:
    """Should raise error when path is a directory."""
    from mcp_base import MCPError

    params = tool.get_params_model()(path=str(tmp_dir))
    with pytest.raises(MCPError, match="not a file"):
        await tool.execute(params)


async def test_encrypt_produces_decryptable_output(
    tool: EncryptFile, plaintext_file: Path
) -> None:
    """Encrypted output should be decryptable with the key."""
    from cryptography.fernet import Fernet

    original_content = plaintext_file.read_bytes()

    params = tool.get_params_model()(path=str(plaintext_file))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    # Decrypt manually to verify
    key = result.data.identity_public_key.encode("utf-8")
    fernet = Fernet(key)
    ciphertext = Path(result.data.path).read_bytes()
    decrypted = fernet.decrypt(ciphertext)

    assert decrypted == original_content


def test_metadata(tool: EncryptFile) -> None:
    """Should have correct tool metadata."""
    assert tool.name == "security.encrypt_file"
    assert tool.confirmation_required is True
    assert tool.undo_supported is False
