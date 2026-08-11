"""Evening briefing: Gmail/Calendar/YouTube via MCP → Telegram."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import ConfigurationError, Settings, get_settings
from app.integrations.telegram import TelegramError, TelegramService
from app.mcp import MCPError
from app.mcp.client import MCPClient

logger = logging.getLogger(__name__)

MAX_EMAILS_IN_BRIEFING = 5
MAX_EVENTS_IN_BRIEFING = 5
MAX_VIDEOS_IN_BRIEFING = 3
MAX_SNIPPET_LEN = 80
MAX_SUBJECT_LEN = 80
MAX_SENDER_LEN = 40
TELEGRAM_MAX_MESSAGE_LEN = 4000


class BriefingError(Exception):
    """Raised when briefing generation or delivery fails."""


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1].rstrip() + "…"


def _display_sender(sender: str) -> str:
    sender = (sender or "").strip()
    if not sender:
        return "Unknown"
    match = re.match(r'^"?([^"<]+)"?\s*<[^>]+>$', sender)
    if match:
        name = match.group(1).strip()
        if name:
            return _truncate(name, MAX_SENDER_LEN)
    return _truncate(sender, MAX_SENDER_LEN)


def _sanitize_plain(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text or "")


def _format_date_heading(date_str: str | None, timezone: str) -> str:
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")

    if date_str:
        try:
            day = datetime.fromisoformat(date_str).date()
            return day.strftime("%A, %d %B")
        except ValueError:
            pass
    return datetime.now(tz).strftime("%A, %d %B")


def format_evening_briefing(
    *,
    gmail_data: dict[str, Any] | None,
    calendar_data: dict[str, Any] | None,
    youtube_data: dict[str, Any] | None,
    timezone: str = "Asia/Kolkata",
) -> str:
    date_heading = _format_date_heading(
        (gmail_data or {}).get("date") or (calendar_data or {}).get("date"),
        timezone,
    )
    lines: list[str] = ["🌙 EVENING BRIEF", date_heading, ""]

    # Gmail
    if gmail_data is not None:
        emails = gmail_data.get("emails") or []
        if not isinstance(emails, list):
            emails = []
        count = gmail_data.get("count", len(emails))
        lines.extend(["📧 Gmail", f"{count} email{'s' if count != 1 else ''}", ""])
        for email in emails[:MAX_EMAILS_IN_BRIEFING]:
            if not isinstance(email, dict):
                continue
            lines.append(f"• {_display_sender(str(email.get('sender') or ''))}")
            lines.append(
                f"  {_truncate(str(email.get('subject') or '(no subject)'), MAX_SUBJECT_LEN)}"
            )
        if emails:
            lines.append("")
    else:
        lines.extend(["📧 Gmail", "Unavailable", ""])

    # Calendar (prefer upcoming / tomorrow-oriented payload)
    if calendar_data is not None:
        events = calendar_data.get("events") or []
        if not isinstance(events, list):
            events = []
        count = calendar_data.get("count", len(events))
        lines.extend(["📅 Calendar", f"{count} meeting{'s' if count != 1 else ''}", ""])
        for event in events[:MAX_EVENTS_IN_BRIEFING]:
            if not isinstance(event, dict):
                continue
            lines.append(f"• {_truncate(str(event.get('title') or '(no title)'), MAX_SUBJECT_LEN)}")
            if event.get("start"):
                lines.append(f"  {event['start']}")
        if events:
            lines.append("")
    else:
        lines.extend(["📅 Calendar", "Unavailable", ""])

    # YouTube
    if youtube_data is not None:
        videos = youtube_data.get("videos") or []
        if not isinstance(videos, list):
            videos = []
        count = youtube_data.get("count", len(videos))
        lines.extend(
            [
                "▶️ YouTube",
                f"{count} new video{'s' if count != 1 else ''} from your selected channels",
                "",
            ]
        )
        for video in videos[:MAX_VIDEOS_IN_BRIEFING]:
            if not isinstance(video, dict):
                continue
            channel = _truncate(str(video.get("channel") or "YouTube"), MAX_SENDER_LEN)
            title = _truncate(str(video.get("title") or "(no title)"), MAX_SUBJECT_LEN)
            lines.append(f"• {channel}")
            lines.append(f"  {title}")
        if youtube_data.get("message") and not videos:
            lines.append(str(youtube_data["message"]))
        if videos or youtube_data.get("message"):
            lines.append("")
    else:
        lines.extend(["▶️ YouTube", "Unavailable", ""])

    # Priority: first upcoming/calendar event if present
    priority_event = None
    if calendar_data and isinstance(calendar_data.get("events"), list) and calendar_data["events"]:
        priority_event = calendar_data["events"][0]
    if isinstance(priority_event, dict):
        lines.append("🔥 Priority")
        title = _truncate(str(priority_event.get("title") or "Event"), MAX_SUBJECT_LEN)
        start = str(priority_event.get("start") or "")
        lines.append(f"{title}" + (f" at {start}" if start else ""))
        lines.append("")

    text = _sanitize_plain("\n".join(lines).rstrip() + "\n")
    if len(text) > TELEGRAM_MAX_MESSAGE_LEN:
        text = text[: TELEGRAM_MAX_MESSAGE_LEN - 1].rstrip() + "…"
    return text


def format_gmail_briefing(
    gmail_data: dict[str, Any],
    *,
    timezone: str = "Asia/Kolkata",
) -> str:
    """Gmail-only briefing formatter (kept for tests / simple callers)."""
    if not isinstance(gmail_data, dict):
        raise BriefingError("Invalid Gmail briefing payload")

    emails = gmail_data.get("emails") or []
    if not isinstance(emails, list):
        emails = []

    count = gmail_data.get("count")
    if not isinstance(count, int):
        count = len(emails)

    date_heading = _format_date_heading(gmail_data.get("date"), timezone)
    lines: list[str] = [
        "🌙 EVENING BRIEF",
        date_heading,
        "",
        "📧 Gmail",
        f"You received {count} email{'s' if count != 1 else ''} today.",
        "",
    ]

    if emails:
        lines.append("🔥 Emails")
        for email in emails[:MAX_EMAILS_IN_BRIEFING]:
            if not isinstance(email, dict):
                continue
            sender = _display_sender(str(email.get("sender") or ""))
            subject = _truncate(str(email.get("subject") or "(no subject)"), MAX_SUBJECT_LEN)
            snippet = _truncate(str(email.get("snippet") or ""), MAX_SNIPPET_LEN)
            lines.append(f"• {sender}")
            lines.append(f"  {subject}")
            if snippet:
                lines.append(f"  {snippet}")
            lines.append("")
    else:
        lines.append("No emails to highlight today.")
        lines.append("")

    lines.append("📊 Summary")
    lines.append(f"{count} email{'s' if count != 1 else ''} received")

    text = _sanitize_plain("\n".join(lines).rstrip() + "\n")
    if len(text) > TELEGRAM_MAX_MESSAGE_LEN:
        text = text[: TELEGRAM_MAX_MESSAGE_LEN - 1].rstrip() + "…"
    return text


async def _safe_tool(
    mcp_client: MCPClient,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        result = await mcp_client.call_tool(tool_name, arguments)
    except MCPError as exc:
        logger.error("Briefing tool %s failed: %s", tool_name, exc)
        return None
    return result if isinstance(result, dict) else None


async def generate_gmail_briefing(
    mcp_client: MCPClient,
    settings: Settings | None = None,
) -> str:
    """Fetch connected services through MCP and format an evening briefing."""
    settings = settings or get_settings()
    gmail = await _safe_tool(mcp_client, "gmail.get_today_emails")
    calendar = await _safe_tool(
        mcp_client, "calendar.get_upcoming_events", {"days": 1}
    )
    youtube = await _safe_tool(
        mcp_client, "youtube.get_recent_videos", {"limit": 5}
    )

    if gmail is None and calendar is None and youtube is None:
        raise BriefingError("All briefing data sources failed")

    return format_evening_briefing(
        gmail_data=gmail,
        calendar_data=calendar,
        youtube_data=youtube,
        timezone=settings.timezone,
    )


async def send_evening_briefing(
    mcp_client: MCPClient,
    settings: Settings | None = None,
) -> str:
    """MCP sources → format briefing → Telegram."""
    settings = settings or get_settings()
    text = await generate_gmail_briefing(mcp_client, settings)

    try:
        telegram = TelegramService.from_settings(settings)
        await telegram.send_message(text)
    except ConfigurationError as exc:
        logger.error("Briefing Telegram configuration error: %s", exc)
        raise BriefingError(str(exc)) from exc
    except TelegramError as exc:
        logger.error("Briefing Telegram send failed: %s", exc)
        raise BriefingError(str(exc)) from exc

    logger.info("Evening briefing sent to Telegram")
    return text
