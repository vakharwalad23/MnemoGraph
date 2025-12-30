"""
Tokenizer module for token counting and content type detection.

Provides accurate token counting using tiktoken with fast approximation fallback.
Used to determine if content should be treated as Note (< 2000 tokens) or
Document (>= 2000 tokens).
"""

from src.config import TokenizerConfig
from src.core.tokenizer.tokenizer import ContentType, Tokenizer

__all__ = ["Tokenizer", "TokenizerConfig", "ContentType"]
