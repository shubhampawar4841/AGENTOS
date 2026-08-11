"""
Provenance guardrails: block claims about external data that no tool verified.

The system prompt is not a sufficient defense, so every candidate final answer
is screened here before it can reach the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SERVICE_TOOLS: dict[str, tuple[str, ...]] = {
    "gmail": ("gmail.get_today_emails",),
    "calendar": ("calendar.get_today_events", "calendar.get_upcoming_events"),
    "youtube": ("youtube.get_recent_videos",),
}

SERVICE_LABELS = {
    "gmail": "Gmail",
    "calendar": "Google Calendar",
    "youtube": "YouTube",
}

_SERVICE_HINTS: dict[str, tuple[str, ...]] = {
    "gmail": (r"\bemails?\b", r"\be-?mails?\b", r"\binbox\b", r"\bgmail\b", r"\bmails?\b"),
    "calendar": (
        r"\bcalendar\b",
        r"\bmeetings?\b",
        r"\bschedule[ds]?\b",
        r"\bevents?\b",
        r"\bappointments?\b",
        r"\binterview\b",
    ),
    "youtube": (r"\byoutube\b", r"\bvideos?\b", r"\buploads?\b", r"\bchannels?\b"),
}

# Phrasings that assert the assistant already observed real user data.
_ACCESS_CLAIM_PATTERNS: tuple[str, ...] = (
    r"\bi(?:'ve|\s+have)?\s+(?:just\s+|already\s+)?"
    r"(?:checked|check|reviewed|reviewing|looked\s+(?:at|through|over)|scanned|read|"
    r"retrieved|fetched|pulled|accessed|analyz(?:ed|ing)|examined|found|"
    r"gone\s+through|went\s+through)\b",
    r"\bi\s+(?:can\s+see|see|noticed|spotted)\b",
    r"\bchecked\s+your\b",
    r"\bhere\s+(?:are|is)\s+(?:your|the|a\s+few|the\s+top)\b",
    r"\bit\s+looks\s+like\s+you\s+(?:have|received|got)\b",
    r"\byou\s+(?:have|had|received|got)\s+(?:\d+|no|one|two|three|four|five|"
    r"a\s+few|several|some|only)\b",
    r"\byour\s+(?:inbox|calendar|schedule)\s+(?:has|shows|contains|includes)\b",
    r"\bthe\s+(?:top|most\s+important)\s+\d+\b",
)

# Honest failure/ability statements must never be treated as data claims.
_HONEST_DISCLAIMERS: tuple[str, ...] = (
    r"\b(?:can'?t|cannot|could\s*n'?t|couldn't|unable\s+to|failed\s+to|"
    r"had\s+trouble|no\s+access)\b[^.]{0,40}\b(?:access|reach|connect|retrieve|read|check)\b",
    r"\breconnect\b",
    r"\bhaven'?t\s+(?:checked|accessed|looked)\b",
)

# Textual function-call syntax the model must never emit as prose.
_RAW_TOOL_SYNTAX = re.compile(
    r"</?function\b"
    r"|<\|python_tag\|>"
    r"|</?tool_call>"
    r"|\bfunctions\.[a-z_]+\s*\("
    r"|\b(?:gmail|calendar|youtube)__[a-z_]+\s*(?:\(|\{|>)",
    re.IGNORECASE,
)

_RAW_TOOL_BLOCK = re.compile(
    r"<function[^>]*>.*?(?:</function>|$)"
    r"|<tool_call>.*?(?:</tool_call>|$)"
    r"|<\|python_tag\|>.*",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ContentVerdict:
    """Outcome of screening one candidate assistant reply."""

    ok: bool
    reason: str | None = None
    unverified_services: tuple[str, ...] = ()
    safe_response: str = ""


def tool_service(tool_name: str) -> str | None:
    for service, tools in SERVICE_TOOLS.items():
        if tool_name in tools:
            return service
    return None


def contains_raw_tool_syntax(text: str) -> bool:
    """True when the model wrote function-call syntax instead of calling a tool."""
    return bool(_RAW_TOOL_SYNTAX.search(text or ""))


def strip_raw_tool_syntax(text: str) -> str:
    """Remove any function-call syntax so it can never reach the user."""
    return _RAW_TOOL_BLOCK.sub("", text or "").strip()


def mentioned_services(text: str) -> tuple[str, ...]:
    lowered = (text or "").lower()
    found = [
        service
        for service, hints in _SERVICE_HINTS.items()
        if any(re.search(hint, lowered) for hint in hints)
    ]
    return tuple(found)


def _has_honest_disclaimer(lowered: str) -> bool:
    return any(re.search(pattern, lowered) for pattern in _HONEST_DISCLAIMERS)


def claims_external_data(text: str) -> bool:
    """True when the text asserts it observed real user data."""
    lowered = (text or "").lower()
    if not lowered:
        return False
    if _has_honest_disclaimer(lowered):
        return False
    return any(re.search(pattern, lowered) for pattern in _ACCESS_CLAIM_PATTERNS)


def unverified_claimed_services(
    text: str,
    verified_services: set[str] | frozenset[str],
) -> tuple[str, ...]:
    """Services the text talks about as observed data without verified provenance."""
    if not claims_external_data(text):
        return ()
    return tuple(
        service
        for service in mentioned_services(text)
        if service not in verified_services
    )


def access_failure_message(services: tuple[str, ...]) -> str:
    if not services:
        return "I had trouble accessing that just now. Please try again."
    labels = [SERVICE_LABELS[service] for service in services if service in SERVICE_LABELS]
    if not labels:
        return "I had trouble accessing that just now. Please try again."
    joined = labels[0] if len(labels) == 1 else " and ".join([", ".join(labels[:-1]), labels[-1]])
    return (
        f"I couldn't access your {joined} right now. "
        "Please reconnect your Google account and try again."
    )


def review_assistant_content(
    content: str,
    verified_services: set[str] | frozenset[str],
) -> ContentVerdict:
    """
    Screen a candidate final reply for fabricated external data.

    Verification is supplied by the caller and only ever reflects real MCP
    executions, never anything the model asserted.
    """
    text = (content or "").strip()
    if contains_raw_tool_syntax(text):
        services = mentioned_services(text)
        return ContentVerdict(
            ok=False,
            reason="raw_tool_syntax",
            unverified_services=services,
            safe_response=(
                f"I had trouble accessing your {SERVICE_LABELS[services[0]]} just now. "
                "Please try again."
                if services and services[0] in SERVICE_LABELS
                else "I had trouble accessing that just now. Please try again."
            ),
        )

    unverified = unverified_claimed_services(text, verified_services)
    if unverified:
        return ContentVerdict(
            ok=False,
            reason="unverified_data_claim",
            unverified_services=unverified,
            safe_response=access_failure_message(unverified),
        )

    return ContentVerdict(ok=True, safe_response=text)
