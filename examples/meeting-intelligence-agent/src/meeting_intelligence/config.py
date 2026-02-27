import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
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
        local_base_url=os.getenv("MIA_LOCAL_BASE_URL", "http://localhost:8080/v1"),
        local_model=os.getenv("MIA_LOCAL_MODEL", "local"),
        local_ctx_size=int(os.getenv("MIA_LOCAL_CTX_SIZE", "32768")),
        local_n_gpu_layers=int(os.getenv("MIA_LOCAL_GPU_LAYERS", "99")),
        max_tokens=int(os.getenv("MIA_MAX_TOKENS", "8192")),
        max_context_messages=int(os.getenv("MIA_MAX_CONTEXT_MESSAGES", "40")),
    )
