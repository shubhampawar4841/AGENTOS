"""
Register (or delete) the Telegram webhook without exposing secrets.

Reads TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET, and PUBLIC_BASE_URL from the
environment/.env so the bot token and secret never appear in shell history.

Usage:
    python scripts/set_telegram_webhook.py            # register the webhook
    python scripts/set_telegram_webhook.py --info     # show current state
    python scripts/set_telegram_webhook.py --delete   # remove the webhook
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx
from dotenv import load_dotenv

WEBHOOK_PATH = "/api/telegram/webhook"


def _require(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        print(f"Missing {name}. Set it in .env or the environment.")
        raise SystemExit(1)
    return value


def _redact_url(url: str) -> str:
    """Show the webhook host/path without the secret path segment."""
    if not url:
        return "(not set)"
    base, _, _ = url.partition(WEBHOOK_PATH)
    return f"{base}{WEBHOOK_PATH}/<secret>" if base != url else url


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Manage the Telegram webhook")
    parser.add_argument("--delete", action="store_true", help="delete the webhook")
    parser.add_argument("--info", action="store_true", help="show webhook info only")
    args = parser.parse_args()

    token = _require("TELEGRAM_BOT_TOKEN")
    api = f"https://api.telegram.org/bot{token}"

    with httpx.Client(timeout=30.0) as client:
        if args.info:
            info = client.get(f"{api}/getWebhookInfo").json().get("result", {})
            print("url:", _redact_url(info.get("url", "")))
            print("pending_update_count:", info.get("pending_update_count"))
            print("last_error_message:", info.get("last_error_message"))
            return 0

        if args.delete:
            data = client.post(
                f"{api}/deleteWebhook",
                json={"drop_pending_updates": False},
            ).json()
            print("deleteWebhook ok:", data.get("ok"), data.get("description", ""))
            return 0 if data.get("ok") else 1

        secret = _require("TELEGRAM_WEBHOOK_SECRET")
        base_url = _require("PUBLIC_BASE_URL").rstrip("/")
        data = client.post(
            f"{api}/setWebhook",
            json={
                "url": f"{base_url}{WEBHOOK_PATH}/{secret}",
                "secret_token": secret,
                "allowed_updates": ["message"],
                "drop_pending_updates": True,
            },
        ).json()

    if not data.get("ok"):
        print("setWebhook failed:", data.get("description", "unknown error"))
        return 1

    print(f"Webhook registered at {base_url}{WEBHOOK_PATH}/<secret>")
    print("Local polling is now bypassed; Telegram delivers updates to the deployment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
