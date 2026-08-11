"""Deterministic Telegram-friendly formatters for agent replies."""

from __future__ import annotations

import re
from typing import Any

MAX_EMAILS = 10
MAX_SNIPPET = 70
MAX_SUBJECT = 70
MAX_SENDER = 36


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def display_sender(sender: str) -> str:
    sender = (sender or "").strip()
    if not sender:
        return "Unknown"
    match = re.match(r'^"?([^"<]+)"?\s*<[^>]+>$', sender)
    if match and match.group(1).strip():
        return _truncate(match.group(1).strip(), MAX_SENDER)
    return _truncate(sender, MAX_SENDER)


def format_today_emails(gmail_data: dict[str, Any]) -> str:
    emails = gmail_data.get("emails") if isinstance(gmail_data, dict) else []
    if not isinstance(emails, list):
        emails = []
    count = gmail_data.get("count") if isinstance(gmail_data, dict) else len(emails)
    if not isinstance(count, int):
        count = len(emails)

    lines = [
        "📧 Today's Gmail",
        "",
        f"You received {count} email{'s' if count != 1 else ''}.",
        "",
    ]

    if not emails:
        lines.append("No emails found for today.")
        return "\n".join(lines)

    # Heuristic: first half (or up to 5) treated as "important" for mobile readability.
    important_n = min(5, max(1, (len(emails) + 1) // 2))
    important = emails[:important_n]
    other = emails[important_n:MAX_EMAILS]

    lines.append("🔥 Important")
    lines.append("")
    for email in important:
        if not isinstance(email, dict):
            continue
        lines.append(f"• {display_sender(str(email.get('sender') or ''))}")
        lines.append(f"  {_truncate(str(email.get('subject') or '(no subject)'), MAX_SUBJECT)}")
        lines.append("")

    if other:
        lines.append("📬 Other")
        for email in other:
            if not isinstance(email, dict):
                continue
            lines.append(
                f"• {display_sender(str(email.get('sender') or ''))} — "
                f"{_truncate(str(email.get('subject') or '(no subject)'), MAX_SUBJECT)}"
            )
        lines.append("")

    omitted = count - min(count, MAX_EMAILS, len(emails))
    if omitted > 0:
        lines.append(f"…and {omitted} more")

    return "\n".join(lines).rstrip() + "\n"


def format_email_summary_deterministic(gmail_data: dict[str, Any]) -> str:
    emails = gmail_data.get("emails") if isinstance(gmail_data, dict) else []
    if not isinstance(emails, list):
        emails = []
    count = gmail_data.get("count") if isinstance(gmail_data, dict) else len(emails)
    if not isinstance(count, int):
        count = len(emails)

    lines = [
        "📧 Email summary",
        "",
        f"You received {count} email{'s' if count != 1 else ''} today.",
        "",
    ]
    if not emails:
        lines.append("Nothing important to highlight.")
        return "\n".join(lines)

    lines.append("Here are the important emails:")
    lines.append("")
    for email in emails[:MAX_EMAILS]:
        if not isinstance(email, dict):
            continue
        sender = display_sender(str(email.get("sender") or ""))
        subject = _truncate(str(email.get("subject") or "(no subject)"), MAX_SUBJECT)
        snippet = _truncate(str(email.get("snippet") or ""), MAX_SNIPPET)
        lines.append(f"• {sender}")
        lines.append(f"  {subject}")
        if snippet:
            lines.append(f"  {snippet}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def emails_context_for_llm(gmail_data: dict[str, Any]) -> str:
    emails = gmail_data.get("emails") if isinstance(gmail_data, dict) else []
    if not isinstance(emails, list):
        emails = []
    rows: list[str] = []
    for email in emails[:MAX_EMAILS]:
        if not isinstance(email, dict):
            continue
        rows.append(
            f"- from: {email.get('sender') or ''}\n"
            f"  subject: {email.get('subject') or ''}\n"
            f"  snippet: {email.get('snippet') or ''}"
        )
    count = gmail_data.get("count", len(emails)) if isinstance(gmail_data, dict) else len(emails)
    return f"date={gmail_data.get('date') if isinstance(gmail_data, dict) else ''}\ncount={count}\n" + "\n".join(rows)


def format_calendar_events(calendar_data: dict[str, Any], *, heading: str = "📅 Calendar") -> str:
    events = calendar_data.get("events") if isinstance(calendar_data, dict) else []
    if not isinstance(events, list):
        events = []
    count = calendar_data.get("count") if isinstance(calendar_data, dict) else len(events)
    if not isinstance(count, int):
        count = len(events)

    lines = [heading, "", f"{count} event{'s' if count != 1 else ''}.", ""]
    if not events:
        lines.append("No events found.")
        return "\n".join(lines)

    for event in events[:MAX_EMAILS]:
        if not isinstance(event, dict):
            continue
        title = _truncate(str(event.get("title") or "(no title)"), MAX_SUBJECT)
        start = str(event.get("start") or "")
        location = _truncate(str(event.get("location") or ""), 40)
        lines.append(f"• {title}")
        if start:
            lines.append(f"  {start}")
        if location:
            lines.append(f"  📍 {location}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_youtube_videos(youtube_data: dict[str, Any]) -> str:
    videos = youtube_data.get("videos") if isinstance(youtube_data, dict) else []
    if not isinstance(videos, list):
        videos = []
    count = youtube_data.get("count") if isinstance(youtube_data, dict) else len(videos)
    if not isinstance(count, int):
        count = len(videos)

    lines = ["▶️ YouTube", "", f"{count} recent video{'s' if count != 1 else ''}.", ""]
    if youtube_data.get("message") and not videos:
        lines.append(str(youtube_data["message"]))
        return "\n".join(lines)

    if not videos:
        lines.append("No recent videos found.")
        return "\n".join(lines)

    for video in videos[:MAX_EMAILS]:
        if not isinstance(video, dict):
            continue
        title = _truncate(str(video.get("title") or "(no title)"), MAX_SUBJECT)
        channel = _truncate(str(video.get("channel") or ""), MAX_SENDER)
        lines.append(f"• {channel}" if channel else "• Video")
        lines.append(f"  {title}")
        url = str(video.get("url") or "")
        if url:
            lines.append(f"  {url}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_overview(
    *,
    gmail_data: dict[str, Any] | None = None,
    calendar_data: dict[str, Any] | None = None,
    youtube_data: dict[str, Any] | None = None,
) -> str:
    lines = ["🧭 Today", ""]
    if gmail_data is not None:
        count = gmail_data.get("count", 0)
        lines.append(f"📧 Gmail — {count} email{'s' if count != 1 else ''}")
    if calendar_data is not None:
        count = calendar_data.get("count", 0)
        lines.append(f"📅 Calendar — {count} event{'s' if count != 1 else ''}")
    if youtube_data is not None:
        count = youtube_data.get("count", 0)
        lines.append(f"▶️ YouTube — {count} video{'s' if count != 1 else ''}")
    lines.append("")

    if gmail_data is not None:
        lines.append(format_today_emails(gmail_data).rstrip())
        lines.append("")
    if calendar_data is not None:
        lines.append(format_calendar_events(calendar_data, heading="📅 Today's calendar").rstrip())
        lines.append("")
    if youtube_data is not None:
        lines.append(format_youtube_videos(youtube_data).rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
