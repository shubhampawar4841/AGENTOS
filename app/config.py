"""Typed application configuration loaded from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_GOOGLE_REDIRECT_URI = "http://localhost:3000/auth/google/callback"
DEFAULT_GOOGLE_TOKEN_PATH = "tokens/google_token.json"
DEFAULT_LLM_PROVIDER = "none"


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    app_env: str
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    telegram_webhook_secret: str | None
    timezone: str
    briefing_time: str
    google_client_id: str | None
    google_client_secret: str | None
    google_redirect_uri: str
    google_token_path: str
    llm_provider: str
    llm_api_key: str | None
    llm_model: str | None
    llm_base_url: str | None
    youtube_channel_ids: tuple[str, ...]

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def is_serverless(self) -> bool:
        """True on Vercel / production where long-running pollers cannot run."""
        if os.getenv("VERCEL", "").strip():
            return True
        return self.app_env.lower() in {"production", "prod"}

    @property
    def telegram_transport(self) -> str:
        """
        How inbound Telegram messages are received.

        Local development uses long polling. Vercel/production uses webhooks.
        """
        if not self.telegram_configured:
            return "disabled"
        if self.is_serverless:
            return "webhook"
        return "polling"

    @property
    def telegram_webhook_configured(self) -> bool:
        return self.telegram_configured and bool(self.telegram_webhook_secret)

    @property
    def google_configured(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)

    @property
    def llm_configured(self) -> bool:
        provider = (self.llm_provider or "none").lower()
        return provider not in {"", "none", "off", "disabled"} and bool(self.llm_api_key)

    def require_telegram(self) -> tuple[str, str]:
        """Return Telegram credentials or raise if missing."""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            raise ConfigurationError(
                "Telegram is not configured. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment."
            )
        return self.telegram_bot_token, self.telegram_chat_id

    def require_google(self) -> tuple[str, str]:
        """Return Google OAuth client credentials or raise if missing."""
        if not self.google_client_id or not self.google_client_secret:
            raise ConfigurationError(
                "Google OAuth is not configured. "
                "Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in the environment."
            )
        return self.google_client_id, self.google_client_secret

    def google_token_file(self) -> Path:
        return Path(self.google_token_path)

    def briefing_hour_minute(self) -> tuple[int, int]:
        """Parse BRIEFING_TIME (HH:MM) into hour and minute."""
        try:
            parts = self.briefing_time.strip().split(":")
            if len(parts) != 2:
                raise ValueError("expected HH:MM")
            hour = int(parts[0])
            minute = int(parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError("hour/minute out of range")
            return hour, minute
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid BRIEFING_TIME '{self.briefing_time}'. "
                "Expected HH:MM (e.g. 19:00)."
            ) from exc


def _optional(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _csv_ids(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings from the environment."""
    settings = Settings(
        app_env=os.getenv("APP_ENV", "development").strip() or "development",
        telegram_bot_token=_optional("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_optional("TELEGRAM_CHAT_ID"),
        telegram_webhook_secret=_optional("TELEGRAM_WEBHOOK_SECRET"),
        timezone=os.getenv("TIMEZONE", "Asia/Kolkata").strip() or "Asia/Kolkata",
        briefing_time=os.getenv("BRIEFING_TIME", "19:00").strip() or "19:00",
        google_client_id=_optional("GOOGLE_CLIENT_ID"),
        google_client_secret=_optional("GOOGLE_CLIENT_SECRET"),
        google_redirect_uri=(
            os.getenv("GOOGLE_REDIRECT_URI", DEFAULT_GOOGLE_REDIRECT_URI).strip()
            or DEFAULT_GOOGLE_REDIRECT_URI
        ),
        google_token_path=(
            os.getenv("GOOGLE_TOKEN_PATH", DEFAULT_GOOGLE_TOKEN_PATH).strip()
            or DEFAULT_GOOGLE_TOKEN_PATH
        ),
        llm_provider=(
            os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
            or DEFAULT_LLM_PROVIDER
        ),
        llm_api_key=_optional("LLM_API_KEY"),
        llm_model=_optional("LLM_MODEL"),
        llm_base_url=_optional("LLM_BASE_URL"),
        youtube_channel_ids=_csv_ids("YOUTUBE_CHANNEL_IDS"),
    )
    # Validate scheduler config eagerly so bad values fail fast at startup.
    settings.briefing_hour_minute()
    logger.debug("Loaded settings for env=%s timezone=%s", settings.app_env, settings.timezone)
    return settings
