from ..config import Config
from .base import LLMClient, LLMResponse
from .llama_client import LlamaClient


def get_llm_client(config: Config) -> LLMClient:
    """Factory: return the local llama.cpp backend."""
    return LlamaClient(config)


__all__ = ["get_llm_client", "LLMClient", "LLMResponse"]
