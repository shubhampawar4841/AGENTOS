"""Telegram Bot API integration."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import ConfigurationError, Settings

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"


class TelegramError(Exception):
    """Raised when the Telegram API returns an error or is unreachable."""


class TelegramService:
    """Thin wrapper around the Telegram Bot API sendMessage method."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        if not bot_token or not chat_id:
            raise ConfigurationError(
                "Telegram is not configured. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment."
            )
        self._bot_token = bot_token
        self._chat_id = chat_id

    @classmethod
    def from_settings(cls, settings: Settings) -> TelegramService:
        token, chat_id = settings.require_telegram()
        return cls(bot_token=token, chat_id=chat_id)

    async def send_message(self, text: str) -> dict[str, Any]:
        """Send a text message to the configured chat."""
        url = f"{TELEGRAM_API_BASE}/bot{self._bot_token}/sendMessage"
        payload = {"chat_id": self._chat_id, "text": text}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            logger.error("Telegram request failed: %s", type(exc).__name__)
            raise TelegramError("Failed to reach Telegram API") from exc

        try:
            data = response.json()
        except ValueError as exc:
            logger.error("Telegram returned non-JSON response (status=%s)", response.status_code)
            raise TelegramError("Invalid response from Telegram API") from exc

        if response.status_code >= 400 or not data.get("ok", False):
            description = data.get("description", "unknown error")
            logger.error(
                "Telegram API error (status=%s): %s",
                response.status_code,
                description,
            )
            raise TelegramError(f"Telegram API error: {description}")

        logger.info("Telegram message sent successfully")
        return data
