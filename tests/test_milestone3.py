"""Milestone 3 tests: Gmail briefing → Telegram workflow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.mcp import MCPError
from app.mcp.client import MCPClient
from app.services.briefing import (
    BriefingError,
    format_gmail_briefing,
    generate_gmail_briefing,
    send_evening_briefing,
)


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
    get_settings.cache_clear()


def test_format_gmail_briefing_with_emails():
    text = format_gmail_briefing(
        {
            "date": "2026-08-11",
            "count": 2,
            "emails": [
                {
                    "id": "1",
                    "sender": "Interview Team <hr@example.com>",
                    "subject": "Interview invitation",
                    "snippet": "We would like to invite you to interview next week.",
                },
                {
                    "id": "2",
                    "sender": "GitHub <noreply@github.com>",
                    "subject": "Pull request review requested",
                    "snippet": "Please review PR #42",
                },
            ],
        },
        timezone="Asia/Kolkata",
    )

    assert "🌙 EVENING BRIEF" in text
    assert "Tuesday, 11 August" in text
    assert "You received 2 emails today." in text
    assert "• Interview Team" in text
    assert "Interview invitation" in text
    assert "• GitHub" in text
    assert "📊 Summary" in text
    assert "2 emails received" in text


def test_format_gmail_briefing_no_emails():
    text = format_gmail_briefing(
        {"date": "2026-08-11", "count": 0, "emails": []},
        timezone="Asia/Kolkata",
    )
    assert "You received 0 emails today." in text
    assert "No emails to highlight today." in text
    assert "0 emails received" in text


def test_format_truncates_long_snippet():
    long_snippet = "x" * 200
    text = format_gmail_briefing(
        {
            "date": "2026-08-11",
            "count": 1,
            "emails": [
                {
                    "sender": "a@example.com",
                    "subject": "Hi",
                    "snippet": long_snippet,
                }
            ],
        }
    )
    assert "x" * 200 not in text
    assert "…" in text


@pytest.mark.asyncio
async def test_generate_gmail_briefing_uses_mcp(settings_env):
    client = MagicMock(spec=MCPClient)
    client.call_tool = AsyncMock(
        return_value={
            "date": "2026-08-11",
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

    text = await generate_gmail_briefing(client, get_settings())
    client.call_tool.assert_awaited_once_with("gmail.get_today_emails")
    assert "• Ada" in text
    assert "Hello" in text


@pytest.mark.asyncio
async def test_generate_gmail_briefing_mcp_error(settings_env):
    client = MagicMock(spec=MCPClient)
    client.call_tool = AsyncMock(side_effect=MCPError("auth required"))

    with pytest.raises(BriefingError, match="auth required"):
        await generate_gmail_briefing(client, get_settings())


@pytest.mark.asyncio
async def test_send_evening_briefing_sends_telegram(settings_env):
    client = MagicMock(spec=MCPClient)
    client.call_tool = AsyncMock(
        return_value={"date": "2026-08-11", "count": 0, "emails": []}
    )

    with patch(
        "app.services.briefing.TelegramService.send_message",
        new_callable=AsyncMock,
        return_value={"ok": True},
    ) as send:
        text = await send_evening_briefing(client, get_settings())

    assert "EVENING BRIEF" in text
    send.assert_awaited_once()
    assert "EVENING BRIEF" in send.await_args.args[0]


def test_test_briefing_endpoint_success(settings_env):
    from app.main import app

    with patch(
        "app.main.send_evening_briefing",
        new_callable=AsyncMock,
        return_value="briefing",
    ):
        with TestClient(app) as client:
            response = client.post("/test/briefing")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "Evening briefing sent to Telegram",
    }


def test_test_briefing_endpoint_failure(settings_env):
    from app.main import app

    with patch(
        "app.main.send_evening_briefing",
        new_callable=AsyncMock,
        side_effect=BriefingError("Gmail failed"),
    ):
        with TestClient(app) as client:
            response = client.post("/test/briefing")

    assert response.status_code == 502
    assert "Gmail failed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_scheduler_job_swallows_briefing_errors(settings_env):
    from app.scheduler.jobs import create_scheduler

    mcp = MagicMock(spec=MCPClient)
    scheduler = create_scheduler(get_settings(), mcp)
    job = scheduler.get_job("evening_briefing")
    assert job is not None

    with patch(
        "app.scheduler.jobs.send_evening_briefing",
        new_callable=AsyncMock,
        side_effect=BriefingError("boom"),
    ):
        # Should not raise
        await job.func()
