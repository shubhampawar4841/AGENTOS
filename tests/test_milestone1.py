"""Basic Milestone 1 tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import ConfigurationError, Settings, get_settings
from app.integrations.telegram import TelegramError, TelegramService
from app.mcp.client import MCPClient
from app.mcp.servers.gmail import GmailMCPServer


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings_env(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    monkeypatch.setenv("TIMEZONE", "Asia/Kolkata")
    monkeypatch.setenv("BRIEFING_TIME", "19:00")
    get_settings.cache_clear()


def test_health_endpoint(settings_env):
    from app.main import app

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_telegram_service_requires_configuration():
    with pytest.raises(ConfigurationError):
        TelegramService(bot_token="", chat_id="123")

    with pytest.raises(ConfigurationError):
        Settings(
            app_env="test",
            telegram_bot_token=None,
            telegram_chat_id=None,
            telegram_webhook_secret=None,
            telegram_mode=None,
            timezone="Asia/Kolkata",
            briefing_time="19:00",
            google_client_id=None,
            google_client_secret=None,
            google_redirect_uri="http://localhost:3000/auth/google/callback",
            google_token_path="tokens/google_token.json",
            google_token_json=None,
            public_base_url=None,
            llm_provider="none",
            llm_api_key=None,
            llm_model=None,
            llm_base_url=None,
            youtube_channel_ids=(),
        ).require_telegram()


def test_telegram_service_from_settings_missing(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    monkeypatch.setenv("TIMEZONE", "Asia/Kolkata")
    monkeypatch.setenv("BRIEFING_TIME", "19:00")
    get_settings.cache_clear()

    settings = get_settings()
    with pytest.raises(ConfigurationError):
        TelegramService.from_settings(settings)


@pytest.mark.asyncio
async def test_telegram_send_message_success(settings_env):
    settings = get_settings()
    service = TelegramService.from_settings(settings)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": True, "result": {"message_id": 1}}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("app.integrations.telegram.httpx.AsyncClient", return_value=mock_client):
        result = await service.send_message("hello")

    assert result["ok"] is True
    mock_client.post.assert_awaited_once()
    args, kwargs = mock_client.post.await_args
    assert args[0].endswith("/sendMessage")
    assert "test-token" in args[0]
    assert kwargs["json"]["chat_id"] == "12345"
    assert kwargs["json"]["text"] == "hello"


@pytest.mark.asyncio
async def test_telegram_send_message_api_error(settings_env):
    settings = get_settings()
    service = TelegramService.from_settings(settings)

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.json.return_value = {"ok": False, "description": "Unauthorized"}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("app.integrations.telegram.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(TelegramError, match="Unauthorized"):
            await service.send_message("hello")


@pytest.mark.asyncio
async def test_telegram_send_message_network_error(settings_env):
    settings = get_settings()
    service = TelegramService.from_settings(settings)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("boom"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("app.integrations.telegram.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(TelegramError, match="Failed to reach Telegram API"):
            await service.send_message("hello")


def test_test_telegram_endpoint_success(settings_env):
    from app.main import app

    with patch.object(
        TelegramService,
        "send_message",
        new_callable=AsyncMock,
        return_value={"ok": True},
    ):
        with TestClient(app) as client:
            response = client.post("/test/telegram")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "test-token" not in response.text


def test_test_telegram_endpoint_missing_config(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-should-not-leak")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setenv("TIMEZONE", "Asia/Kolkata")
    monkeypatch.setenv("BRIEFING_TIME", "19:00")
    get_settings.cache_clear()

    from app.main import app

    with TestClient(app) as client:
        response = client.post("/test/telegram")

    assert response.status_code == 503
    assert "secret-should-not-leak" not in response.text


@pytest.mark.asyncio
async def test_mcp_gmail_tool_with_mocked_service():
    mock_service = MagicMock()
    mock_service.get_today_emails.return_value = {
        "date": "2026-08-11",
        "count": 1,
        "emails": [
            {
                "id": "1",
                "thread_id": "t1",
                "sender": "a@example.com",
                "subject": "Hi",
                "snippet": "Hello",
                "timestamp": "2026-08-11T10:00:00",
                "labels": ["INBOX"],
            }
        ],
    }
    server = GmailMCPServer(gmail_service=mock_service)
    tools = server.list_tools()
    assert any(t.name == "gmail.get_today_emails" for t in tools)

    result = await server.get_today_emails()
    assert result["success"] is True
    assert result["count"] == 1
    assert result["emails"][0]["sender"] == "a@example.com"


@pytest.mark.asyncio
async def test_mcp_client_discover_and_call():
    mock_service = MagicMock()
    mock_service.get_today_emails.return_value = {
        "date": "2026-08-11",
        "count": 0,
        "emails": [],
    }
    client = MCPClient()
    client.register_server(GmailMCPServer(gmail_service=mock_service))
    names = [t.name for t in client.list_tools()]
    assert "gmail.get_today_emails" in names

    result = await client.call_tool("gmail.get_today_emails")
    assert result["success"] is True
    assert "emails" in result


@pytest.mark.asyncio
async def test_mcp_client_unknown_tool():
    client = MCPClient()
    client.register_server(GmailMCPServer())
    from app.mcp import MCPError

    with pytest.raises(MCPError):
        await client.call_tool("gmail.does_not_exist")


def test_invalid_briefing_time(monkeypatch):
    monkeypatch.setenv("BRIEFING_TIME", "25:99")
    monkeypatch.setenv("TIMEZONE", "Asia/Kolkata")
    get_settings.cache_clear()
    with pytest.raises(ConfigurationError, match="BRIEFING_TIME"):
        get_settings()
