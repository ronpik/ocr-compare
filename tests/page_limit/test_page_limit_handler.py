import pytest
from typing import Any
from ocrtool.page_limit.page_limit_handler import PageLimitHandler
from ocrtool.page_limit.exceptions import PageLimitExceededError

class DummyExecutor:
    pass

def test_is_page_limit_error_detects_known_strings() -> None:
    """Test that is_page_limit_error returns True for known error messages."""
    handler = PageLimitHandler()
    for msg in handler.PAGE_LIMIT_ERROR_STRINGS:
        exc = Exception(f"Some error: {msg}")
        assert handler.is_page_limit_error(exc)

def test_is_page_limit_error_returns_false_for_unrelated_error() -> None:
    """Test that is_page_limit_error returns False for unrelated errors."""
    handler = PageLimitHandler()
    exc = Exception("Some other error")
    assert not handler.is_page_limit_error(exc)

def test_handle_raises_custom_exception() -> None:
    """Test that handle raises PageLimitExceededError."""
    handler = PageLimitHandler()
    exc = Exception("PAGE_LIMIT_EXCEEDED: too many pages")
    with pytest.raises(PageLimitExceededError):
        handler.handle(exc, b"", DummyExecutor()) 