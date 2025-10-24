"""
LLM provider abstraction layer for text generation.

Supported providers:
- Ollama (native SDK)
- OpenAI (official SDK)
"""
from src.core.llm.base import LLMProvider
from src.core.llm.ollama import OllamaLLM
from src.core.llm.openai import OpenAILLM

__all__ = [
    "LLMProvider",
    "OllamaLLM",
    "OpenAILLM",
]

