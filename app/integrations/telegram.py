"""Telegram Bot API integration: send + long-poll receive."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.config import ConfigurationError, Settings

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
TELEGRAM_MAX_MESSAGE_LEN = 4096
SAFE_CHUNK_LEN = 3900


class TelegramError(Exception):
    """Raised when the Telegram API returns an error or is unreachable."""


MessageHandler = Callable[[str], Awaitable[str]]


def split_telegram_message(text: str, limit: int = SAFE_CHUNK_LEN) -> list[str]:
    """Split a long reply into Telegram-safe chunks."""
    text = (text or "").strip()
    if not text:
        return [""]
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip("\n")
    return chunks


class TelegramService:
    """Thin wrapper around the Telegram Bot API."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        if not bot_token or not chat_id:
            raise ConfigurationError(
                "Telegram is not configured. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment."
            )
        self._bot_token = bot_token
        self._chat_id = str(chat_id)

    @classmethod
    def from_settings(cls, settings: Settings) -> TelegramService:
        token, chat_id = settings.require_telegram()
        return cls(bot_token=token, chat_id=chat_id)

    @property
    def allowed_chat_id(self) -> str:
        return self._chat_id

    def _url(self, method: str) -> str:
        return f"{TELEGRAM_API_BASE}/bot{self._bot_token}/{method}"

    async def _post(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(self._url(method), json=payload or {})
        except httpx.TimeoutException as exc:
            logger.error("Telegram %s timed out", method)
            raise TelegramError(f"Telegram {method} timed out") from exc
        except httpx.HTTPError as exc:
            logger.error("Telegram %s request failed: %s", method, type(exc).__name__)
            raise TelegramError(f"Failed to reach Telegram API ({method})") from exc

        try:
            data = response.json()
        except ValueError as exc:
            logger.error(
                "Telegram %s returned non-JSON response (status=%s)",
                method,
                response.status_code,
            )
            raise TelegramError("Invalid response from Telegram API") from exc

        if response.status_code == 429 or (
            isinstance(data, dict) and data.get("error_code") == 429
        ):
            retry_after = 1
            if isinstance(data, dict):
                retry_after = int((data.get("parameters") or {}).get("retry_after", 1))
            raise TelegramError(f"Telegram rate limited; retry_after={retry_after}")

        if response.status_code >= 400 or not data.get("ok", False):
            description = data.get("description", "unknown error")
            logger.error(
                "Telegram API error on %s (status=%s): %s",
                method,
                response.status_code,
                description,
            )
            raise TelegramError(f"Telegram API error: {description}")

        return data

    async def send_message(
        self,
        text: str,
        *,
        chat_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a text message (auto-splits if longer than Telegram's limit)."""
        target = str(chat_id or self._chat_id)
        last: dict[str, Any] = {}
        for chunk in split_telegram_message(text):
            last = await self._post(
                "sendMessage",
                {"chat_id": target, "text": chunk},
            )
        logger.info("Telegram message sent successfully")
        return last

    async def get_updates(
        self,
        *,
        offset: int | None = None,
        timeout: int = 25,
    ) -> list[dict[str, Any]]:
        """Long-poll Telegram getUpdates."""
        payload: dict[str, Any] = {
            "timeout": timeout,
            "allowed_updates": ["message"],
        }
        if offset is not None:
            payload["offset"] = offset

        # Long poll: HTTP timeout must exceed Telegram's timeout.
        data = await self._post("getUpdates", payload, timeout=float(timeout + 10))
        result = data.get("result") or []
        if not isinstance(result, list):
            raise TelegramError("Invalid getUpdates payload")
        return result


class TelegramPoller:
    """
    Background long-polling loop for inbound Telegram messages.

    Only messages from the configured TELEGRAM_CHAT_ID are processed.
    Offset is kept in memory (no database).
    """

    def __init__(
        self,
        telegram: TelegramService,
        message_handler: MessageHandler,
        *,
        poll_timeout: int = 25,
    ) -> None:
        self._telegram = telegram
        self._handler = message_handler
        self._poll_timeout = poll_timeout
        self._offset: int | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._backoff = 1.0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            logger.warning("Telegram poller already running; not starting a second task")
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="telegram-poller")
        logger.info("Telegram poller started")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("Telegram poller stopped")

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                updates = await self._telegram.get_updates(
                    offset=self._offset,
                    timeout=self._poll_timeout,
                )
                self._backoff = 1.0
                for update in updates:
                    await self._handle_update(update)
            except asyncio.CancelledError:
                raise
            except TelegramError as exc:
                message = str(exc)
                logger.error("Telegram poller error: %s", message)
                if "retry_after=" in message:
                    try:
                        wait = float(message.rsplit("=", 1)[-1])
                    except ValueError:
                        wait = self._backoff
                    await asyncio.sleep(max(wait, 1.0))
                else:
                    await asyncio.sleep(self._backoff)
                    self._backoff = min(self._backoff * 2, 60.0)
            except Exception:
                logger.exception("Telegram poller unexpected failure")
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, 60.0)

    async def _handle_update(self, update: dict[str, Any]) -> None:
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            self._offset = update_id + 1

        message = update.get("message") or {}
        if not isinstance(message, dict):
            return

        chat = message.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        if chat_id != self._telegram.allowed_chat_id:
            logger.warning("Ignoring Telegram message from unauthorized chat_id")
            return

        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            return

        try:
            reply = await self._handler(text.strip())
        except Exception:
            logger.exception("Agent failed while handling Telegram message")
            reply = "⚠️ Something went wrong while processing that. Please try again."

        try:
            await self._telegram.send_message(reply, chat_id=chat_id)
        except TelegramError as exc:
            logger.error("Failed to send Telegram reply: %s", exc)
