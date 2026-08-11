"""Telegram webhook transport tests for Vercel/production."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import ConfigurationError, get_settings
from app.integrations.telegram import (
    TelegramService,
    handle_update,
    validate_webhook_secret,
)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings_env(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "test")
    # Ignore the developer's local .env transport choice.
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("TELEGRAM_MODE", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "webhook-secret-xyz")
    monkeypatch.setenv("TIMEZONE", "Asia/Kolkata")
    monkeypatch.setenv("BRIEFING_TIME", "19:00")
    monkeypatch.setenv("GOOGLE_TOKEN_PATH", str(tmp_path / "token.json"))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    get_settings.cache_clear()


@pytest.fixture
def production_env(settings_env, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("VERCEL", "1")
    get_settings.cache_clear()


def test_validate_webhook_secret_accepts_path_or_header():
    assert validate_webhook_secret("abc", path_secret="abc") is True
    assert validate_webhook_secret("abc", header_secret="abc") is True
    assert validate_webhook_secret("abc", path_secret="nope", header_secret="abc") is True
    assert validate_webhook_secret("abc", path_secret="nope") is False
    assert validate_webhook_secret(None, path_secret="abc") is False


def test_transport_defaults_to_webhook_outside_local_dev(settings_env):
    # APP_ENV=test is not a local dev env, so polling must not be inferred.
    settings = get_settings()
    assert settings.telegram_transport == "webhook"
    assert settings.polling_enabled is False


def test_local_development_uses_polling(settings_env, monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    get_settings.cache_clear()
    assert get_settings().telegram_transport == "polling"


def test_production_and_vercel_use_webhook(settings_env, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()
    assert get_settings().telegram_transport == "webhook"

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("VERCEL", "1")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.is_serverless is True
    assert settings.telegram_transport == "webhook"
    assert settings.polling_enabled is False


def test_explicit_telegram_mode_overrides_environment(settings_env, monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("TELEGRAM_MODE", "webhook")
    get_settings.cache_clear()
    assert get_settings().telegram_transport == "webhook"

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("TELEGRAM_MODE", "polling")
    get_settings.cache_clear()
    assert get_settings().telegram_transport == "polling"


def test_invalid_telegram_mode_fails_fast(settings_env, monkeypatch):
    monkeypatch.setenv("TELEGRAM_MODE", "sockets")
    get_settings.cache_clear()
    with pytest.raises(ConfigurationError):
        get_settings()


def test_transport_disabled_without_telegram_credentials(settings_env, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    get_settings.cache_clear()
    assert get_settings().telegram_transport == "disabled"


def test_root_endpoint(settings_env):
    from app.main import app

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "SYNCOS"}


@pytest.mark.asyncio
async def test_shared_handle_update_processes_allowed_chat():
    telegram = TelegramService(bot_token="t", chat_id="12345")
    telegram.send_message = AsyncMock(return_value={"ok": True})
    handler = AsyncMock(return_value="Hey 👋 What's up?")

    result = await handle_update(
        {
            "update_id": 7,
            "message": {"chat": {"id": 12345}, "text": "hey"},
        },
        telegram=telegram,
        message_handler=handler,
    )

    assert result["handled"] is True
    handler.assert_awaited_once_with("hey", "12345")
    telegram.send_message.assert_awaited_once_with(
        "Hey 👋 What's up?",
        chat_id="12345",
    )


@pytest.mark.asyncio
async def test_shared_handle_update_ignores_other_chats():
    telegram = TelegramService(bot_token="t", chat_id="12345")
    telegram.send_message = AsyncMock(return_value={"ok": True})
    handler = AsyncMock(return_value="ok")

    result = await handle_update(
        {
            "update_id": 8,
            "message": {"chat": {"id": 999}, "text": "hey"},
        },
        telegram=telegram,
        message_handler=handler,
    )

    assert result == {
        "handled": False,
        "reason": "unauthorized_chat",
        "update_id": 8,
    }
    handler.assert_not_awaited()
    telegram.send_message.assert_not_awaited()


def test_webhook_status_endpoint(production_env):
    from app.main import app

    with TestClient(app) as client:
        response = client.get("/telegram/webhook/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is True
    assert payload["mode"] == "webhook"
    assert payload["webhook_secret_configured"] is True
    assert "test-token" not in response.text
    assert "webhook-secret-xyz" not in response.text


def test_production_does_not_start_poller(production_env):
    from app.main import app

    with TestClient(app) as client:
        assert client.app.state.telegram_poller is None
        assert client.app.state.telegram is not None
        status = client.get("/telegram/webhook/status").json()
        assert status["mode"] == "webhook"


def test_webhook_requires_configured_secret(production_env, monkeypatch):
    from app.main import app

    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)
    get_settings.cache_clear()

    with TestClient(app) as client:
        response = client.post(
            "/api/telegram/webhook",
            json={"update_id": 1, "message": {"chat": {"id": 12345}, "text": "hey"}},
        )

    assert response.status_code == 503


def test_webhook_rejects_bad_secret(production_env):
    from app.main import app

    with TestClient(app) as client:
        response = client.post(
            "/api/telegram/webhook/wrong-secret",
            json={
                "update_id": 1,
                "message": {"chat": {"id": 12345}, "text": "hey"},
            },
        )

    assert response.status_code == 403


def test_webhook_accepts_path_secret_and_handles_update(production_env):
    from app.main import app

    with patch(
        "app.main.handle_update",
        new_callable=AsyncMock,
        return_value={"handled": True, "update_id": 99},
    ) as mocked:
        with TestClient(app) as client:
            response = client.post(
                "/api/telegram/webhook/webhook-secret-xyz",
                json={
                    "update_id": 99,
                    "message": {"chat": {"id": 12345}, "text": "hey"},
                },
            )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["handled"] is True
    mocked.assert_awaited_once()


def test_webhook_accepts_header_secret(production_env):
    from app.main import app

    with patch(
        "app.main.handle_update",
        new_callable=AsyncMock,
        return_value={"handled": True, "update_id": 100},
    ):
        with TestClient(app) as client:
            response = client.post(
                "/api/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": "webhook-secret-xyz"},
                json={
                    "update_id": 100,
                    "message": {"chat": {"id": 12345}, "text": "hey"},
                },
            )

    assert response.status_code == 200
    assert response.json()["ok"] is True
