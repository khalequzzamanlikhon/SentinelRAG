"""
Input validation and content safety guardrails for SentinelRAG.

Performs multi-layered checks on user queries before they enter the
pipeline, preventing prompt injection, excessively long inputs, and
other edge cases.

NEW in v2.1.
"""

import re
import logging
from typing import Optional

from src.config import settings
from src.state import GuardrailResult
from src.exceptions import GuardrailViolationError

logger = logging.getLogger(__name__)


def validate_query(query: str) -> GuardrailResult:
    """Run all guardrail checks on a user query.

    Args:
        query: The raw user query string.

    Returns:
        A ``GuardrailResult`` with ``passed=True`` if all checks pass.

    Raises:
        GuardrailViolationError: If guardrails are enabled and the query fails.
    """
    if not settings.enable_guardrails:
        return GuardrailResult(passed=True, reason="Guardrails disabled.")

    # Check 1: Empty query
    if not query or not query.strip():
        raise GuardrailViolationError(
            "Query is empty.",
            details={"check": "empty_query"},
        )

    # Check 2: Max length
    if len(query) > settings.max_query_length:
        raise GuardrailViolationError(
            f"Query exceeds maximum length of {settings.max_query_length} characters.",
            details={
                "check": "max_length",
                "query_length": len(query),
                "max_length": settings.max_query_length,
            },
        )

    # Check 3: Blocked patterns (prompt injection attempts)
    query_lower = query.lower()
    for pattern in settings.blocked_patterns:
        if pattern.lower() in query_lower:
            raise GuardrailViolationError(
                f"Query contains a blocked pattern.",
                details={
                    "check": "blocked_pattern",
                    "pattern": pattern,
                },
            )

    # Check 4: Unicode homoglyph detection (basic)
    suspicious_chars = _detect_homoglyphs(query)
    if suspicious_chars:
        raise GuardrailViolationError(
            "Query contains suspicious Unicode characters.",
            details={
                "check": "homoglyph_detection",
                "suspicious_chars": suspicious_chars,
            },
        )

    logger.debug("Guardrail checks passed for query (len=%d).", len(query))
    return GuardrailResult(passed=True, reason="All checks passed.")


def _detect_homoglyphs(text: str) -> list[str]:
    """Detect potentially suspicious Unicode homoglyph characters.

    Checks for characters that visually resemble ASCII but are from
    different Unicode blocks (common in prompt injection attacks).

    Returns:
        List of suspicious character strings found.
    """
    suspicious: list[str] = []
    for char in text:
        code_point = ord(char)
        # Cyrillic 'а' (U+0430) looks like Latin 'a' (U+0061)
        # Full-width Latin letters (U+FF21-U+FF3A, U+FF41-U+FF5A)
        if 0x0400 <= code_point <= 0x04FF:  # Cyrillic block
            suspicious.append(f"U+{code_point:04X}")
        elif 0xFF01 <= code_point <= 0xFF5E:  # Fullwidth block
            suspicious.append(f"U+{code_point:04X}")
        elif 0x200B <= code_point <= 0x200F:  # Zero-width characters
            suspicious.append(f"U+{code_point:04X}")
        elif 0x2028 <= code_point <= 0x2029:  # Line/paragraph separators
            suspicious.append(f"U+{code_point:04X}")

    return suspicious


def sanitize_query(query: str) -> str:
    """Lightly sanitize a query by stripping extra whitespace and control chars.

    Args:
        query: The raw query string.

    Returns:
        Cleaned query string.
    """
    # Strip leading/trailing whitespace
    cleaned = query.strip()

    # Remove control characters (except common whitespace)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)

    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned
