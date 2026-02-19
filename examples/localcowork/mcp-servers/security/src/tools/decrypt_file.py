"""
security.decrypt_file — Decrypt an encrypted file.

Reads the encrypted file and its corresponding .key file (located
at encrypted_path + .key or encrypted_path with .enc replaced by .key),
decrypts the content, and writes the plaintext to output_path.
Mutable operation: confirmation required.
"""

from __future__ import annotations

from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes
from validation import assert_absolute_path, assert_sandboxed


# ─── Params / Result Models ────────────────────────────────────────────────


class Params(BaseModel):
    """Parameters for security.decrypt_file."""

    path: str = Field(description="Absolute path to the encrypted file")
    output_path: str | None = Field(
        default=None,
        description="Output path for the decrypted file. Defaults to path without .enc",
    )


class Result(BaseModel):
    """Return value for security.decrypt_file."""

    path: str = Field(description="Path to the decrypted file")


# ─── Tool Implementation ───────────────────────────────────────────────────


class DecryptFile(MCPTool[Params, Result]):
    """Decrypt an encrypted file using its Fernet key."""

    name = "security.decrypt_file"
    description = "Decrypt an encrypted file using its Fernet key"
    confirmation_required = True
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Decrypt the specified file."""
        assert_absolute_path(params.path, "path")
        assert_sandboxed(params.path)

        encrypted_path = Path(params.path)
        if not encrypted_path.exists():
            raise MCPError(ErrorCodes.FILE_NOT_FOUND, f"File not found: {params.path}")
        if not encrypted_path.is_file():
            raise MCPError(ErrorCodes.INVALID_PARAMS, f"Path is not a file: {params.path}")

        # Locate key file
        key_path = _find_key_file(params.path)
        if key_path is None:
            raise MCPError(
                ErrorCodes.FILE_NOT_FOUND,
                f"Key file not found for: {params.path}. "
                f"Expected at {params.path}.key or with .key extension.",
            )

        # Read key and ciphertext
        try:
            key = Path(key_path).read_bytes()
        except (OSError, PermissionError) as e:
            raise MCPError(ErrorCodes.INTERNAL_ERROR, f"Failed to read key file: {e}") from e

        try:
            ciphertext = encrypted_path.read_bytes()
        except (OSError, PermissionError) as e:
            raise MCPError(
                ErrorCodes.INTERNAL_ERROR, f"Failed to read encrypted file: {e}"
            ) from e

        # Decrypt
        try:
            fernet = Fernet(key)
            plaintext = fernet.decrypt(ciphertext)
        except InvalidToken as e:
            raise MCPError(
                ErrorCodes.INTERNAL_ERROR,
                "Decryption failed: invalid key or corrupted data",
            ) from e

        # Write decrypted output
        output_path = _resolve_output_path(params.path, params.output_path)
        assert_sandboxed(output_path)

        try:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(plaintext)
        except (OSError, PermissionError) as e:
            raise MCPError(
                ErrorCodes.INTERNAL_ERROR, f"Failed to write decrypted file: {e}"
            ) from e

        return MCPResult(success=True, data=Result(path=str(output)))


# ─── Helper Functions ──────────────────────────────────────────────────────


def _find_key_file(encrypted_path: str) -> str | None:
    """Locate the key file for an encrypted file."""
    # Try path + .key (e.g., file.txt.enc.key)
    candidate1 = Path(encrypted_path + ".key")
    if candidate1.exists():
        return str(candidate1)

    # Try replacing .enc with .key (e.g., file.txt.key)
    if encrypted_path.endswith(".enc"):
        candidate2 = Path(encrypted_path[:-4] + ".key")
        if candidate2.exists():
            return str(candidate2)

    return None


def _resolve_output_path(source_path: str, output_path: str | None) -> str:
    """Resolve the output path for decryption."""
    if output_path is not None:
        assert_absolute_path(output_path, "output_path")
        return output_path

    # Strip .enc extension if present
    if source_path.endswith(".enc"):
        return source_path[:-4]

    return source_path + ".decrypted"
