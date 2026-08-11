"""Simple intent/tool router for the conversational agent."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any


class Intent(str, Enum):
    START = "start"
    GMAIL_TODAY = "gmail_today"
    GMAIL_SUMMARY = "gmail_summary"
    CALENDAR_TODAY = "calendar_today"
    CALENDAR_UPCOMING = "calendar_upcoming"
    YOUTUBE = "youtube"
    OVERVIEW_TODAY = "overview_today"
    UNKNOWN = "unknown"


# Explicit allow-list of tools the conversational agent may invoke.
ALLOWED_AGENT_TOOLS = frozenset(
    {
        "gmail.get_today_emails",
        "calendar.get_today_events",
        "calendar.get_upcoming_events",
        "youtube.get_recent_videos",
    }
)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


_GMAIL_TODAY_PATTERNS = (
    r"\bemails?\b",
    r"\bgmail\b",
    r"\binbox\b",
    r"what.*(got|received|came).*(mail|email)",
    r"show.*(today|emails)",
    r"any.*(mail|email)",
)

_GMAIL_SUMMARY_PATTERNS = (
    r"\bsummar.*(mail|email)",
    r"important.*(mail|email)",
    r"highlight.*(mail|email)",
)

_CALENDAR_TODAY_PATTERNS = (
    r"(meeting|calendar|schedule|event).*(today|tonight)",
    r"(today|tonight).*(meeting|calendar|schedule|event)",
    r"what('?s| is)? on my calendar today",
    r"what meetings do i have today",
)

_CALENDAR_UPCOMING_PATTERNS = (
    r"\btomorrow\b",
    r"coming up",
    r"this week",
    r"upcoming",
    r"what do i have",
    r"what('?s| is)? (on )?(my )?(schedule|calendar)",
    r"meetings?",
)

_YOUTUBE_PATTERNS = (
    r"\byoutube\b",
    r"\bvideos?\b",
    r"favorite channels?",
    r"uploaded",
)

_OVERVIEW_PATTERNS = (
    r"what happened today",
    r"what('?s| is)? going on today",
    r"today('?s)? (overview|update|digest|brief)",
)


def detect_intent(message: str) -> Intent:
    text = (message or "").strip().lower()
    if not text:
        return Intent.UNKNOWN

    if text.startswith("/start") or text in {"start", "hi", "hello", "help", "/help"}:
        return Intent.START

    if any(re.search(p, text) for p in _OVERVIEW_PATTERNS):
        return Intent.OVERVIEW_TODAY

    if any(re.search(p, text) for p in _YOUTUBE_PATTERNS):
        return Intent.YOUTUBE

    if any(re.search(p, text) for p in _CALENDAR_TODAY_PATTERNS):
        return Intent.CALENDAR_TODAY

    if any(re.search(p, text) for p in _CALENDAR_UPCOMING_PATTERNS):
        return Intent.CALENDAR_UPCOMING

    if any(re.search(p, text) for p in _GMAIL_SUMMARY_PATTERNS):
        return Intent.GMAIL_SUMMARY

    if any(re.search(p, text) for p in _GMAIL_TODAY_PATTERNS):
        return Intent.GMAIL_TODAY

    if "email" in text or "mail" in text:
        return Intent.GMAIL_TODAY

    return Intent.UNKNOWN


def plan_tool_calls(message: str) -> list[ToolCall]:
    """
    Deterministic tool planner (used when LLM tool selection is unavailable).

    Never invents tools outside ALLOWED_AGENT_TOOLS.
    """
    intent = detect_intent(message)
    text = (message or "").strip().lower()

    if intent is Intent.START or intent is Intent.UNKNOWN:
        return []

    if intent is Intent.GMAIL_TODAY or intent is Intent.GMAIL_SUMMARY:
        return [ToolCall("gmail.get_today_emails", {})]

    if intent is Intent.CALENDAR_TODAY:
        return [ToolCall("calendar.get_today_events", {})]

    if intent is Intent.CALENDAR_UPCOMING:
        days = 1 if "tomorrow" in text else 7 if "week" in text else 7
        if "tomorrow" in text:
            days = 2  # include remainder of today + tomorrow window via upcoming
        return [ToolCall("calendar.get_upcoming_events", {"days": days})]

    if intent is Intent.YOUTUBE:
        return [ToolCall("youtube.get_recent_videos", {"limit": 10})]

    if intent is Intent.OVERVIEW_TODAY:
        return [
            ToolCall("gmail.get_today_emails", {}),
            ToolCall("calendar.get_today_events", {}),
            ToolCall("youtube.get_recent_videos", {"limit": 5}),
        ]

    return []
