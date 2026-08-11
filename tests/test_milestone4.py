"""Milestone 4 tests: Telegram polling + conversational agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.agent.agent import PersonalAgent, process_message
from app.agent.formatters import format_today_emails
from app.agent.prompts import START_MESSAGE, UNKNOWN_MESSAGE
from app.agent.router import Intent, detect_intent
from app.config import get_settings
from app.integrations.telegram import TelegramPoller, TelegramService, split_telegram_message
from app.mcp import MCPError
from app.mcp.client import MCPClient
from app.services.llm import LLMService


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings_env(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    monkeypatch.setenv("TIMEZONE", "Asia/Kolkata")
    monkeypatch.setenv("BRIEFING_TIME", "19:00")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("GOOGLE_TOKEN_PATH", str(tmp_path / "google_token.json"))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    get_settings.cache_clear()


def test_detect_intent_start():
    assert detect_intent("/start") is Intent.START
    assert detect_intent("hello") is Intent.START


def test_detect_intent_gmail():
    assert detect_intent("What emails did I get today?") is Intent.GMAIL_TODAY
    assert detect_intent("Show today's emails") is Intent.GMAIL_TODAY
    assert detect_intent("Summarize my emails") is Intent.GMAIL_SUMMARY
    assert detect_intent("Any important emails?") is Intent.GMAIL_SUMMARY


def test_detect_intent_unknown():
    assert detect_intent("What is the weather?") is Intent.UNKNOWN


def test_detect_intent_calendar_youtube():
    assert detect_intent("What meetings do I have today?") is Intent.CALENDAR_TODAY
    assert detect_intent("What meetings do I have tomorrow?") is Intent.CALENDAR_UPCOMING
    assert detect_intent("Any new YouTube videos?") is Intent.YOUTUBE
    assert detect_intent("What happened today?") is Intent.OVERVIEW_TODAY


def test_format_today_emails():
    text = format_today_emails(
        {
            "count": 2,
            "emails": [
                {
                    "sender": "QuantumLoopAI <q@example.com>",
                    "subject": "Interview invitation",
                    "snippet": "Hi",
                },
                {
                    "sender": "GitHub <noreply@github.com>",
                    "subject": "PR review requested",
                },
            ],
        }
    )
    assert "📧 Today's Gmail" in text
    assert "You received 2 emails." in text
    assert "QuantumLoopAI" in text


@pytest.mark.asyncio
async def test_agent_start(settings_env):
    agent = PersonalAgent(MagicMock(spec=MCPClient), LLMService(get_settings()))
    reply = await agent.process_message("/start")
    assert reply == START_MESSAGE


@pytest.mark.asyncio
async def test_agent_gmail_uses_mcp(settings_env):
    mcp = MagicMock(spec=MCPClient)
    mcp.call_tool = AsyncMock(
        return_value={
            "count": 1,
            "emails": [
                {
                    "sender": "Ada <ada@example.com>",
                    "subject": "Hello",
                    "snippet": "World",
                }
            ],
        }
    )
    agent = PersonalAgent(mcp, LLMService(get_settings()))
    reply = await agent.process_message("What emails did I get today?")
    mcp.call_tool.assert_awaited_once_with("gmail.get_today_emails", {})
    assert "Ada" in reply
    assert "Hello" in reply


@pytest.mark.asyncio
async def test_agent_unknown(settings_env):
    agent = PersonalAgent(MagicMock(spec=MCPClient), LLMService(get_settings()))
    reply = await agent.process_message("What is the weather?")
    assert reply == UNKNOWN_MESSAGE


@pytest.mark.asyncio
async def test_agent_mcp_auth_error_message(settings_env):
    mcp = MagicMock(spec=MCPClient)
    mcp.call_tool = AsyncMock(
        side_effect=MCPError("Google account is not connected. Open /auth/google")
    )
    agent = PersonalAgent(mcp, LLMService(get_settings()))
    reply = await process_message("Show today's emails", agent)
    assert "Gmail is not connected" in reply or "auth/google" in reply


def test_split_telegram_message():
    chunks = split_telegram_message("a" * 5000, limit=1000)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


@pytest.mark.asyncio
async def test_poller_ignores_other_chat(settings_env):
    telegram = TelegramService(bot_token="t", chat_id="12345")
    handler = AsyncMock(return_value="ok")
    poller = TelegramPoller(telegram, handler)

    await poller._handle_update(
        {
            "update_id": 10,
            "message": {"chat": {"id": 999}, "text": "hi"},
        }
    )
    handler.assert_not_awaited()
    assert poller._offset == 11


@pytest.mark.asyncio
async def test_poller_handles_allowed_chat(settings_env):
    telegram = TelegramService(bot_token="t", chat_id="12345")
    telegram.send_message = AsyncMock(return_value={"ok": True})
    handler = AsyncMock(return_value="reply-text")
    poller = TelegramPoller(telegram, handler)

    await poller._handle_update(
        {
            "update_id": 42,
            "message": {"chat": {"id": 12345}, "text": "What emails did I get today?"},
        }
    )
    handler.assert_awaited_once_with(
        "What emails did I get today?",
        "12345",
    )
    telegram.send_message.assert_awaited_once()
    assert poller._offset == 43


def test_test_agent_endpoint(settings_env):
    from app.main import app

    with patch(
        "app.main.process_message",
        new_callable=AsyncMock,
        return_value="📧 Today's Gmail\n",
    ):
        with TestClient(app) as client:
            response = client.post("/test/agent", json={"message": "/start"})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "Gmail" in response.json()["reply"]
