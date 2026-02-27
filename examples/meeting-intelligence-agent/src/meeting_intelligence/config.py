import os
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Which backend to use
    backend: Literal["anthropic", "local"] = "anthropic"

    # Anthropic settings
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # Local llama.cpp server settings
    local_base_url: str = "http://localhost:8080/v1"
    local_model: str = "local"
    local_api_key: str = "sk-no-key"  # llama.cpp server ignores this
    local_ctx_size: int = 32768
    local_n_gpu_layers: int = 99

    # Agent behavior
    max_tokens: int = 8192
    max_context_messages: int = 40  # before compaction triggers


def load_config() -> Config:
    """Load config from environment variables."""
    return Config(
        backend=os.getenv("MIA_BACKEND", "anthropic"),  # type: ignore[arg-type]
        anthropic_model=os.getenv("MIA_ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        local_base_url=os.getenv("MIA_LOCAL_BASE_URL", "http://localhost:8080/v1"),
        local_model=os.getenv("MIA_LOCAL_MODEL", "local"),
        local_ctx_size=int(os.getenv("MIA_LOCAL_CTX_SIZE", "32768")),
        local_n_gpu_layers=int(os.getenv("MIA_LOCAL_GPU_LAYERS", "99")),
        max_tokens=int(os.getenv("MIA_MAX_TOKENS", "8192")),
        max_context_messages=int(os.getenv("MIA_MAX_CONTEXT_MESSAGES", "40")),
    )
