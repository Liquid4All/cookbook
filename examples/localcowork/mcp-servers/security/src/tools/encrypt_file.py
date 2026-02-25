"""
security.encrypt_file — Encrypt a file using Fernet symmetric encryption.

Reads the source file, generates a Fernet key, encrypts the content,
and writes both the encrypted file and the key file. The key file is
written alongside the encrypted output with a .key extension.
Mutable operation: confirmation required.
"""

from __future__ import annotations

from pathlib import Path

from cryptography.fernet import Fernet
from pydantic import BaseModel, Field

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes
from validation import assert_absolute_path, assert_sandboxed


# ─── Params / Result Models ────────────────────────────────────────────────


class Params(BaseModel):
    """Parameters for security.encrypt_file."""

    path: str = Field(description="Absolute path to the file to encrypt")
    output_path: str | None = Field(
        default=None,
        description="Output path for the encrypted file. Defaults to path + .enc",
    )


class Result(BaseModel):
    """Return value for security.encrypt_file."""

    path: str = Field(description="Path to the encrypted file")
    identity_public_key: str = Field(description="Base64-encoded Fernet key")


# ─── Tool Implementation ───────────────────────────────────────────────────


class EncryptFile(MCPTool[Params, Result]):
    """Encrypt a file using Fernet symmetric encryption."""

    name = "security.encrypt_file"
    description = "Encrypt a file using Fernet symmetric encryption"
    confirmation_required = True
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Encrypt the specified file."""
        assert_absolute_path(params.path, "path")
        assert_sandboxed(params.path)

        source = Path(params.path)
        if not source.exists():
            raise MCPError(ErrorCodes.FILE_NOT_FOUND, f"File not found: {params.path}")
        if not source.is_file():
            raise MCPError(ErrorCodes.INVALID_PARAMS, f"Path is not a file: {params.path}")

        # Determine output path
        output_path = _resolve_output_path(params.path, params.output_path)
        assert_sandboxed(output_path)

        # Generate Fernet key and encrypt
        key = Fernet.generate_key()
        fernet = Fernet(key)

        try:
            plaintext = source.read_bytes()
        except (OSError, PermissionError) as e:
            raise MCPError(ErrorCodes.INTERNAL_ERROR, f"Failed to read file: {e}") from e

        ciphertext = fernet.encrypt(plaintext)

        # Write encrypted file
        output = Path(output_path)
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(ciphertext)
        except (OSError, PermissionError) as e:
            raise MCPError(ErrorCodes.INTERNAL_ERROR, f"Failed to write encrypted file: {e}") from e

        # Write key file alongside encrypted file
        key_path = Path(output_path + ".key")
        try:
            key_path.write_bytes(key)
        except (OSError, PermissionError) as e:
            raise MCPError(ErrorCodes.INTERNAL_ERROR, f"Failed to write key file: {e}") from e

        return MCPResult(
            success=True,
            data=Result(
                path=str(output),
                identity_public_key=key.decode("utf-8"),
            ),
        )


# ─── Helper Functions ──────────────────────────────────────────────────────


def _resolve_output_path(source_path: str, output_path: str | None) -> str:
    """Resolve the output path for encryption."""
    if output_path is not None:
        assert_absolute_path(output_path, "output_path")
        return output_path
    return source_path + ".enc"
