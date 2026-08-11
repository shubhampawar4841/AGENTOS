"""
Discover your Telegram chat ID and write it into .env.

Usage:
    1. Open Telegram, find your bot, and send it any message (e.g. "hi").
    2. Run:  PYTHONPATH=. python scripts/get_telegram_chat_id.py

Never prints the bot token.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import httpx

from app.config import get_settings

ENV_PATH = Path(".env")


def _write_chat_id(chat_id: int | str) -> bool:
    if not ENV_PATH.exists():
        print(f"! {ENV_PATH} not found; set TELEGRAM_CHAT_ID={chat_id} manually.")
        return False

    original = ENV_PATH.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"^TELEGRAM_CHAT_ID=.*$",
        f"TELEGRAM_CHAT_ID={chat_id}",
        original,
        count=1,
        flags=re.MULTILINE,
    )
    if count == 0:
        updated = original.rstrip("\n") + f"\nTELEGRAM_CHAT_ID={chat_id}\n"

    ENV_PATH.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    settings = get_settings()
    token = settings.telegram_bot_token
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set in .env")
        return 1

    response = httpx.get(
        f"https://api.telegram.org/bot{token}/getUpdates", timeout=30.0
    )
    data = response.json()
    if not data.get("ok"):
        print(f"Telegram API error: {data.get('description')}")
        return 1

    chats: dict[int, str] = {}
    for update in data.get("result") or []:
        message = update.get("message") or update.get("edited_message") or {}
        chat = message.get("chat") or {}
        if chat.get("id") is not None:
            label = chat.get("username") or chat.get("title") or chat.get("first_name") or ""
            chats[chat["id"]] = f"{chat.get('type')} {label}".strip()

    if not chats:
        print("No messages found yet.")
        print("Open Telegram, send your bot any message, then re-run this script.")
        return 1

    for chat_id, label in chats.items():
        print(f"found chat: {chat_id}  ({label})")

    chat_id = next(iter(chats))
    if len(chats) > 1:
        print(f"\nMultiple chats found; using the first one: {chat_id}")

    if _write_chat_id(chat_id):
        print(f"\nWrote TELEGRAM_CHAT_ID={chat_id} to .env")
        print("Restart the server, then: curl -X POST https://agentos-rosy.vercel.app/test/telegram")
    return 0


if __name__ == "__main__":
    sys.exit(main())
