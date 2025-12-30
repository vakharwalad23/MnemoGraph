"""
Tests for Tokenizer class.

Tests cover:
1. Token counting (accurate and approximate)
2. Content type detection (Note vs Document)
3. Async threshold detection
4. Tokenization and detokenization
5. Edge cases (empty, unicode, large text)
"""

import pytest

from src.config import TokenizerConfig
from src.core.tokenizer import ContentType, Tokenizer


class TestTokenCounting:
    """Tests for token counting functionality."""

    def test_count_tokens_simple(self):
        """Test basic token counting."""
        tokenizer = Tokenizer()
        count = tokenizer.count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_empty(self):
        """Test counting tokens in empty string."""
        tokenizer = Tokenizer()
        assert tokenizer.count_tokens("") == 0

    def test_count_tokens_whitespace(self):
        """Test counting tokens in whitespace."""
        tokenizer = Tokenizer()
        count = tokenizer.count_tokens("   ")
        assert count > 0  # Whitespace has tokens

    def test_count_tokens_unicode(self):
        """Test counting tokens with unicode characters."""
        tokenizer = Tokenizer()
        count = tokenizer.count_tokens("Hello, 世界! 🌍")
        assert count > 0

    def test_count_tokens_long_text(self):
        """Test counting tokens in long text."""
        tokenizer = Tokenizer()
        long_text = "This is a test sentence. " * 1000
        count = tokenizer.count_tokens(long_text)
        assert count > 5000  # Should be many tokens

    def test_count_tokens_deterministic(self):
        """Test token counting is deterministic."""
        tokenizer = Tokenizer()
        text = "The quick brown fox jumps over the lazy dog."
        count1 = tokenizer.count_tokens(text)
        count2 = tokenizer.count_tokens(text)
        assert count1 == count2


class TestTokenEstimation:
    """Tests for approximate token estimation."""

    def test_estimate_tokens_simple(self):
        """Test basic token estimation."""
        tokenizer = Tokenizer()
        text = "Hello world"
        estimate = tokenizer.estimate_tokens(text)
        assert estimate == len(text) // 4  # Default 4 chars per token

    def test_estimate_tokens_empty(self):
        """Test estimating tokens in empty string."""
        tokenizer = Tokenizer()
        assert tokenizer.estimate_tokens("") == 0

    def test_estimate_tokens_custom_ratio(self):
        """Test estimation with custom chars_per_token."""
        config = TokenizerConfig(chars_per_token=5.0)
        tokenizer = Tokenizer(config)
        text = "Hello world test"  # 16 chars
        estimate = tokenizer.estimate_tokens(text)
        assert estimate == 3  # 16 / 5 = 3.2 -> 3

    def test_approximate_provider(self):
        """Test count_tokens uses estimate when provider is 'approximate'."""
        config = TokenizerConfig(provider="approximate")
        tokenizer = Tokenizer(config)
        text = "Hello world"
        count = tokenizer.count_tokens(text)
        estimate = tokenizer.estimate_tokens(text)
        assert count == estimate


class TestContentTypeDetection:
    """Tests for Note vs Document detection."""

    def test_is_note_short_text(self):
        """Test short text is detected as Note."""
        tokenizer = Tokenizer()
        short_text = "This is a short note."
        assert tokenizer.is_note(short_text) is True
        assert tokenizer.is_document(short_text) is False

    def test_is_document_long_text(self):
        """Test long text is detected as Document."""
        config = TokenizerConfig(document_threshold=100)  # Low threshold for test
        tokenizer = Tokenizer(config)
        long_text = "word " * 200  # ~200 tokens
        assert tokenizer.is_document(long_text) is True
        assert tokenizer.is_note(long_text) is False

    def test_is_document_empty(self):
        """Test empty text is not a document."""
        tokenizer = Tokenizer()
        assert tokenizer.is_document("") is False
        assert tokenizer.is_note("") is True

    def test_detect_content_type_note(self):
        """Test detect_content_type returns NOTE for short text."""
        tokenizer = Tokenizer()
        result = tokenizer.detect_content_type("Short note")
        assert result == ContentType.NOTE

    def test_detect_content_type_document(self):
        """Test detect_content_type returns DOCUMENT for long text."""
        config = TokenizerConfig(document_threshold=50)
        tokenizer = Tokenizer(config)
        long_text = "word " * 100
        result = tokenizer.detect_content_type(long_text)
        assert result == ContentType.DOCUMENT

    def test_threshold_boundary(self):
        """Test detection at exact threshold boundary."""
        config = TokenizerConfig(document_threshold=10)
        tokenizer = Tokenizer(config)

        # Create text with exactly 10 tokens (approximately)
        text = "one two three four five six seven eight nine ten"
        token_count = tokenizer.count_tokens(text)

        # Adjust text to be exactly at threshold
        if token_count >= 10:
            assert tokenizer.is_document(text) is True
        else:
            assert tokenizer.is_note(text) is True


class TestAsyncThreshold:
    """Tests for async processing threshold."""

    def test_requires_async_small_text(self):
        """Test small text doesn't require async."""
        tokenizer = Tokenizer()
        assert tokenizer.requires_async("Short text") is False

    def test_requires_async_large_text(self):
        """Test large text requires async processing."""
        config = TokenizerConfig(async_threshold=100)
        tokenizer = Tokenizer(config)
        large_text = "word " * 200
        assert tokenizer.requires_async(large_text) is True

    def test_requires_async_empty(self):
        """Test empty text doesn't require async."""
        tokenizer = Tokenizer()
        assert tokenizer.requires_async("") is False


class TestTokenization:
    """Tests for tokenize/detokenize operations."""

    def test_tokenize_simple(self):
        """Test basic tokenization."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Hello world")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_tokenize_empty(self):
        """Test tokenizing empty string."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("")
        assert tokens == []

    def test_detokenize_simple(self):
        """Test basic detokenization."""
        tokenizer = Tokenizer()
        original = "Hello world"
        tokens = tokenizer.tokenize(original)
        restored = tokenizer.detokenize(tokens)
        assert restored == original

    def test_detokenize_empty(self):
        """Test detokenizing empty list."""
        tokenizer = Tokenizer()
        assert tokenizer.detokenize([]) == ""

    def test_roundtrip(self):
        """Test tokenize -> detokenize roundtrip preserves text."""
        tokenizer = Tokenizer()
        texts = [
            "Simple text",
            "Hello, 世界!",
            "Multiple\nlines\nhere",
            "Special chars: @#$%^&*()",
        ]
        for text in texts:
            tokens = tokenizer.tokenize(text)
            restored = tokenizer.detokenize(tokens)
            assert restored == text


class TestConfiguration:
    """Tests for tokenizer configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        tokenizer = Tokenizer()
        assert tokenizer.config.provider == "tiktoken"
        assert tokenizer.config.model == "cl100k_base"
        assert tokenizer.config.chars_per_token == 4.0
        assert tokenizer.config.document_threshold == 2000
        assert tokenizer.config.async_threshold == 10000

    def test_custom_config(self):
        """Test custom configuration."""
        config = TokenizerConfig(
            provider="approximate",
            chars_per_token=3.5,
            document_threshold=1000,
            async_threshold=5000,
        )
        tokenizer = Tokenizer(config)
        assert tokenizer.config.provider == "approximate"
        assert tokenizer.config.chars_per_token == 3.5
        assert tokenizer.config.document_threshold == 1000
        assert tokenizer.config.async_threshold == 5000

    def test_lazy_encoder_loading(self):
        """Test encoder is loaded lazily."""
        tokenizer = Tokenizer()
        assert tokenizer._encoder is None
        _ = tokenizer.encoder
        assert tokenizer._encoder is not None

    def test_encoder_reuse(self):
        """Test encoder is reused across calls."""
        tokenizer = Tokenizer()
        encoder1 = tokenizer.encoder
        encoder2 = tokenizer.encoder
        assert encoder1 is encoder2


class TestConfigValidation:
    """Tests for TokenizerConfig validation."""

    def test_chars_per_token_must_be_positive(self):
        """Test chars_per_token must be > 0."""
        with pytest.raises(ValueError):
            TokenizerConfig(chars_per_token=0)

        with pytest.raises(ValueError):
            TokenizerConfig(chars_per_token=-1)

    def test_document_threshold_must_be_positive(self):
        """Test document_threshold must be > 0."""
        with pytest.raises(ValueError):
            TokenizerConfig(document_threshold=0)

    def test_async_threshold_must_be_positive(self):
        """Test async_threshold must be > 0."""
        with pytest.raises(ValueError):
            TokenizerConfig(async_threshold=0)
