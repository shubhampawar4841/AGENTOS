"""Read-only Google Calendar API integration."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.config import Settings, get_settings
from app.integrations.google_auth import (
    GoogleAuthError,
    GoogleAuthRequiredError,
    get_valid_credentials,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 25
DEFAULT_UPCOMING_DAYS = 7


class CalendarError(Exception):
    """Raised when the Calendar API fails or returns unexpected data."""


def _local_tz(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise CalendarError(
            f"Invalid TIMEZONE '{timezone_name}' for Calendar query."
        ) from exc


def _truncate(text: str, limit: int = 240) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def normalize_calendar_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a raw Calendar API event into an agent-friendly shape."""
    if not isinstance(event, dict):
        return None

    status = str(event.get("status") or "confirmed")
    if status == "cancelled":
        return None

    start_raw = event.get("start") or {}
    end_raw = event.get("end") or {}
    if not isinstance(start_raw, dict) or not isinstance(end_raw, dict):
        return None

    start = str(start_raw.get("dateTime") or start_raw.get("date") or "")
    end = str(end_raw.get("dateTime") or end_raw.get("date") or "")
    if not start:
        return None

    return {
        "id": str(event.get("id") or ""),
        "title": str(event.get("summary") or "(no title)"),
        "start": start,
        "end": end,
        "location": str(event.get("location") or ""),
        "description": _truncate(str(event.get("description") or "")),
        "status": status,
        "all_day": "date" in start_raw and "dateTime" not in start_raw,
    }


class CalendarService:
    """Thin read-only wrapper around the Google Calendar API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _day_bounds(self, day: date) -> tuple[datetime, datetime, str]:
        tz = _local_tz(self.settings.timezone)
        start = datetime(day.year, day.month, day.day, tzinfo=tz)
        end = start + timedelta(days=1)
        return start, end, day.isoformat()

    def _list_events(
        self,
        *,
        time_min: datetime,
        time_max: datetime,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> list[dict[str, Any]]:
        try:
            credentials = get_valid_credentials(self.settings)
        except (GoogleAuthRequiredError, GoogleAuthError):
            raise

        try:
            service = build("calendar", "v3", credentials=credentials, cache_discovery=False)
            result = (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=time_min.isoformat(),
                    timeMax=time_max.isoformat(),
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
        except HttpError as exc:
            status = getattr(exc.resp, "status", None)
            logger.error("Calendar list API failed (status=%s)", status)
            if status in {401, 403}:
                raise GoogleAuthRequiredError(
                    "Calendar access was denied. Open /auth/google to authenticate again."
                ) from exc
            raise CalendarError("Calendar API failed while listing events.") from exc
        except (GoogleAuthRequiredError, GoogleAuthError, CalendarError):
            raise
        except Exception as exc:
            logger.error("Unexpected Calendar list failure: %s", type(exc).__name__)
            raise CalendarError("Unexpected error while contacting Google Calendar.") from exc

        if not isinstance(result, dict):
            raise CalendarError("Invalid Calendar list response.")

        raw_events = result.get("items") or []
        if not isinstance(raw_events, list):
            raise CalendarError("Invalid Calendar list response.")

        events: list[dict[str, Any]] = []
        for item in raw_events:
            normalized = normalize_calendar_event(item)
            if normalized is not None:
                events.append(normalized)
        return events

    def get_today_events(self, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
        tz = _local_tz(self.settings.timezone)
        today = datetime.now(tz).date()
        start, end, day = self._day_bounds(today)
        logger.info("Fetching today's Calendar events (limit=%s)", max_results)
        events = self._list_events(time_min=start, time_max=end, max_results=max_results)
        return {"date": day, "count": len(events), "events": events}

    def get_upcoming_events(
        self,
        days: int = DEFAULT_UPCOMING_DAYS,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> dict[str, Any]:
        days = max(1, min(int(days or DEFAULT_UPCOMING_DAYS), 30))
        tz = _local_tz(self.settings.timezone)
        now = datetime.now(tz)
        end = now + timedelta(days=days)
        logger.info("Fetching upcoming Calendar events (days=%s limit=%s)", days, max_results)
        events = self._list_events(time_min=now, time_max=end, max_results=max_results)
        return {
            "from": now.isoformat(),
            "to": end.isoformat(),
            "days": days,
            "count": len(events),
            "events": events,
        }
