"""
Mock embedding module for the knowledge server.

Generates deterministic 128-dimensional float vectors from text using
a hash-based approach.  This is **not** a real embedding model — it
produces vectors that are consistent for identical inputs but carry no
semantic meaning.  The real ``sentence-transformers`` integration will
replace ``generate_embedding`` in a future workstream.

Also provides cosine-similarity and binary (de)serialisation helpers.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Final

# ─── Constants ────────────────────────────────────────────────────────────────

EMBEDDING_DIM: Final[int] = 128
_PACK_FMT: Final[str] = f"<{EMBEDDING_DIM}f"  # little-endian, 128 floats


# ─── Embedding generation ───────────────────────────────────────────────────


def generate_embedding(text: str) -> list[float]:
    """
    Generate a deterministic 128-dim float vector from *text*.

    Strategy: hash the text with SHA-512, expand the digest to fill 128
    float slots (each byte -> value in [-1, 1]), then L2-normalise.
    """
    digest = hashlib.sha512(text.encode("utf-8")).digest()

    # SHA-512 gives 64 bytes; repeat to get 128 values
    raw_bytes = digest + digest

    # Map each byte to [-1.0, 1.0]
    vec = [(b / 127.5) - 1.0 for b in raw_bytes[:EMBEDDING_DIM]]

    # L2-normalise so cosine similarity is a dot product
    return _l2_normalise(vec)


# ─── Similarity ──────────────────────────────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two equal-length vectors.

    Returns a float in [-1, 1].  Both vectors are assumed to have the
    same dimensionality.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vectors must have equal length (got {len(a)} and {len(b)})"
        )

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


# ─── Serialisation ───────────────────────────────────────────────────────────


def serialize_embedding(embedding: list[float]) -> bytes:
    """Pack a float list into a compact binary representation (little-endian)."""
    return struct.pack(_PACK_FMT, *embedding)


def deserialize_embedding(data: bytes) -> list[float]:
    """Unpack bytes produced by ``serialize_embedding`` back to a float list."""
    return list(struct.unpack(_PACK_FMT, data))


# ─── Internal helpers ────────────────────────────────────────────────────────


def _l2_normalise(vec: list[float]) -> list[float]:
    """Return the L2-normalised version of *vec*."""
    magnitude = math.sqrt(sum(v * v for v in vec))
    if magnitude == 0.0:
        return vec
    return [v / magnitude for v in vec]
