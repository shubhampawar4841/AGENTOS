"""Read-only Gmail API integration."""

from __future__ import annotations

import logging
from datetime import datetime
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


class GmailError(Exception):
    """Raised when the Gmail API fails or returns unexpected data."""


def _header_value(headers: list[dict[str, str]], name: str) -> str:
    target = name.lower()
    for header in headers:
        if str(header.get("name", "")).lower() == target:
            return str(header.get("value") or "")
    return ""


def normalize_gmail_message(message: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw Gmail API message into an agent-friendly shape."""
    if not isinstance(message, dict):
        raise GmailError("Invalid Gmail message payload")

    payload = message.get("payload") or {}
    headers = payload.get("headers") if isinstance(payload, dict) else None
    if not isinstance(headers, list):
        headers = []

    internal_date = message.get("internalDate")
    timestamp = ""
    if internal_date is not None:
        try:
            timestamp = datetime.fromtimestamp(int(internal_date) / 1000.0).isoformat()
        except (TypeError, ValueError, OSError):
            timestamp = str(internal_date)

    labels = message.get("labelIds") or []
    if not isinstance(labels, list):
        labels = []

    return {
        "id": str(message.get("id") or ""),
        "thread_id": str(message.get("threadId") or ""),
        "sender": _header_value(headers, "From"),
        "subject": _header_value(headers, "Subject"),
        "snippet": str(message.get("snippet") or ""),
        "timestamp": timestamp,
        "labels": [str(label) for label in labels],
    }


class GmailService:
    """Thin read-only wrapper around the Gmail API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @classmethod
    def from_settings(cls, settings: Settings) -> GmailService:
        return cls(settings=settings)

    def _today_query(self) -> tuple[str, str]:
        try:
            tz = ZoneInfo(self.settings.timezone)
        except ZoneInfoNotFoundError as exc:
            raise GmailError(
                f"Invalid TIMEZONE '{self.settings.timezone}' for Gmail query."
            ) from exc

        today = datetime.now(tz).date()
        # Gmail `after:` is exclusive of the given day in some cases; use epoch seconds.
        start = datetime(today.year, today.month, today.day, tzinfo=tz)
        query = f"after:{int(start.timestamp())}"
        return query, today.isoformat()

    def get_today_emails(self, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
        """
        Fetch a limited set of emails received since local midnight today.

        Returns normalized messages only — never raw Gmail API payloads.
        """
        if max_results < 1:
            max_results = DEFAULT_MAX_RESULTS

        try:
            credentials = get_valid_credentials(self.settings)
        except (GoogleAuthRequiredError, GoogleAuthError):
            raise

        query, today = self._today_query()
        logger.info("Fetching today's Gmail messages (limit=%s)", max_results)

        try:
            service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
            listing = (
                service.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_results)
                .execute()
            )
        except HttpError as exc:
            status = getattr(exc.resp, "status", None)
            logger.error("Gmail list API failed (status=%s)", status)
            if status in {401, 403}:
                raise GoogleAuthRequiredError(
                    "Gmail access was denied. Open /auth/google to authenticate again."
                ) from exc
            raise GmailError("Gmail API failed while listing today's emails.") from exc
        except (GoogleAuthRequiredError, GoogleAuthError, GmailError):
            raise
        except Exception as exc:
            logger.error("Unexpected Gmail list failure: %s", type(exc).__name__)
            raise GmailError("Unexpected error while contacting Gmail.") from exc

        if not isinstance(listing, dict):
            raise GmailError("Invalid Gmail list response.")

        raw_messages = listing.get("messages") or []
        if not isinstance(raw_messages, list):
            raise GmailError("Invalid Gmail list response.")

        emails: list[dict[str, Any]] = []
        for item in raw_messages:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            try:
                full = (
                    service.users()
                    .messages()
                    .get(
                        userId="me",
                        id=item["id"],
                        format="metadata",
                        metadataHeaders=["From", "Subject", "Date"],
                    )
                    .execute()
                )
                emails.append(normalize_gmail_message(full))
            except HttpError as exc:
                logger.error("Gmail get API failed for a message (status=%s)", getattr(exc.resp, "status", None))
                raise GmailError("Gmail API failed while fetching an email.") from exc
            except GmailError:
                raise
            except Exception as exc:
                logger.error("Failed to normalize Gmail message: %s", type(exc).__name__)
                raise GmailError("Failed to parse a Gmail message.") from exc

        logger.info("Retrieved %s Gmail message(s) for %s", len(emails), today)
        return {
            "date": today,
            "count": len(emails),
            "emails": emails,
        }
