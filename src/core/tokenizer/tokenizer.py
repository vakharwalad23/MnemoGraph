"""
Token counting utilities for content type detection.

Uses tiktoken for accurate OpenAI-compatible token counting with
character-based approximation as fallback.
"""

from enum import Enum

import tiktoken

from src.config import TokenizerConfig


class ContentType(str, Enum):
    """Content type based on token count."""

    NOTE = "NOTE"
    DOCUMENT = "DOCUMENT"


class Tokenizer:
    """
    Universal token counter with content type detection.

    Provides accurate token counting using tiktoken.

    Usage:
        tokenizer = Tokenizer()
        count = tokenizer.count_tokens("Hello world")
        is_doc = tokenizer.is_document("Long text...")
        content_type = tokenizer.detect_content_type("Some text")
    """

    def __init__(self, config: TokenizerConfig | None = None):
        """
        Initialize tokenizer with configuration.

        Args:
            config: Optional tokenizer configuration. Uses defaults if not provided.
        """
        self.config = config or TokenizerConfig()
        self._encoder: tiktoken.Encoding | None = None

    @property
    def encoder(self) -> tiktoken.Encoding:
        """
        Lazy-load tiktoken encoder.

        Returns:
            Tiktoken encoding instance
        """
        if self._encoder is None:
            self._encoder = tiktoken.get_encoding(self.config.model)
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """
        Count tokens accurately using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Exact token count
        """
        if not text:
            return 0

        if self.config.provider == "approximate":
            return self.estimate_tokens(text)

        return len(self.encoder.encode(text))

    def estimate_tokens(self, text: str) -> int:
        """
        Fast approximate token count using character ratio.

        Uses the configured chars_per_token ratio (default 4.0) for
        quick estimation without loading the tokenizer.

        Args:
            text: Text to estimate tokens for

        Returns:
            Approximate token count
        """
        if not text:
            return 0
        return int(len(text) / self.config.chars_per_token)

    def is_document(self, text: str) -> bool:
        """
        Check if content should be treated as a Document.

        Uses two-stage detection:
        1. Fast estimate to quickly reject small content
        2. Accurate count only if estimate is near threshold

        Args:
            text: Content to check

        Returns:
            True if content is >= document_threshold tokens
        """
        if not text:
            return False

        threshold = self.config.document_threshold

        # Fast path: if estimate is well below threshold, it's a Note
        estimate = self.estimate_tokens(text)
        if estimate < threshold * 0.8:
            return False

        # Near threshold: use accurate count
        return self.count_tokens(text) >= threshold

    def is_note(self, text: str) -> bool:
        """
        Check if content should be treated as a Note.

        Args:
            text: Content to check

        Returns:
            True if content is < document_threshold tokens
        """
        return not self.is_document(text)

    def requires_async(self, text: str) -> bool:
        """
        Check if content requires async processing.

        Large documents (>= async_threshold) should be queued for
        background processing to avoid blocking.

        Args:
            text: Content to check

        Returns:
            True if content is >= async_threshold tokens
        """
        if not text:
            return False

        threshold = self.config.async_threshold

        # Fast path
        estimate = self.estimate_tokens(text)
        if estimate < threshold * 0.8:
            return False

        return self.count_tokens(text) >= threshold

    def detect_content_type(self, text: str) -> ContentType:
        """
        Detect content type based on token count.

        Args:
            text: Content to classify

        Returns:
            ContentType.NOTE or ContentType.DOCUMENT
        """
        if self.is_document(text):
            return ContentType.DOCUMENT
        return ContentType.NOTE

    def tokenize(self, text: str) -> list[int]:
        """
        Get token IDs for text.

        Useful for chunking operations that need exact token boundaries.

        Args:
            text: Text to tokenize

        Returns:
            List of token IDs
        """
        if not text:
            return []
        return self.encoder.encode(text)

    def detokenize(self, tokens: list[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        if not tokens:
            return ""
        return self.encoder.decode(tokens)
