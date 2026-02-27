from ..config import Config
from .base import LLMClient, LLMResponse
from .anthropic_client import AnthropicClient
from .llama_client import LlamaClient


def get_llm_client(config: Config) -> LLMClient:
    """Factory: return the right backend based on config.backend."""
    if config.backend == "anthropic":
        return AnthropicClient(config)
    elif config.backend == "llama":
        return LlamaClient(config)
    else:
        raise ValueError(f"Unknown backend: {config.backend!r}")


__all__ = ["get_llm_client", "LLMClient", "LLMResponse"]
