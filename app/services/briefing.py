"""Evening briefing: Gmail via MCP → deterministic format → Telegram."""

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

MAX_EMAILS_IN_BRIEFING = 8
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
    """Prefer display name from 'Name <email@x.com>', else the raw sender."""
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
    """Strip control characters; keep plain text (no Telegram parse_mode)."""
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


def format_gmail_briefing(
    gmail_data: dict[str, Any],
    *,
    timezone: str = "Asia/Kolkata",
) -> str:
    """
    Deterministically format today's Gmail MCP payload into a Telegram briefing.

    No LLM — plain text only.
    """
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

        omitted = count - min(count, MAX_EMAILS_IN_BRIEFING, len(emails))
        if omitted > 0:
            lines.append(f"…and {omitted} more")
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


async def generate_gmail_briefing(
    mcp_client: MCPClient,
    settings: Settings | None = None,
) -> str:
    """Fetch today's emails through MCP and format a briefing."""
    settings = settings or get_settings()
    try:
        result = await mcp_client.call_tool("gmail.get_today_emails")
    except MCPError as exc:
        logger.error("Briefing Gmail MCP call failed: %s", exc)
        raise BriefingError(str(exc)) from exc

    if not isinstance(result, dict):
        raise BriefingError("Gmail MCP returned an unexpected payload")

    return format_gmail_briefing(result, timezone=settings.timezone)


async def send_evening_briefing(
    mcp_client: MCPClient,
    settings: Settings | None = None,
) -> str:
    """
    Full vertical slice:
        MCP Gmail → format briefing → Telegram
    """
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
