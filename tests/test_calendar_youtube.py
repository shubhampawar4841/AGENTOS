"""Calendar + YouTube MCP integration tests (mocked Google APIs)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.agent import PersonalAgent
from app.agent.formatters import format_calendar_events, format_youtube_videos
from app.agent.router import Intent, detect_intent, plan_tool_calls
from app.config import get_settings
from app.integrations.calendar import CalendarService, normalize_calendar_event
from app.integrations.google_auth import (
    CALENDAR_READONLY_SCOPE,
    GMAIL_READONLY_SCOPE,
    SCOPES,
    YOUTUBE_READONLY_SCOPE,
    missing_scopes,
)
from app.integrations.youtube import YouTubeService, normalize_youtube_video
from app.mcp.client import create_default_mcp_client
from app.mcp.servers.calendar import CalendarMCPServer
from app.mcp.servers.youtube import YouTubeMCPServer
from app.services.briefing import format_evening_briefing
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
    monkeypatch.setenv("YOUTUBE_CHANNEL_IDS", "UCaaaa,UCbbbb")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    get_settings.cache_clear()


def test_oauth_scopes_include_calendar_and_youtube():
    assert GMAIL_READONLY_SCOPE in SCOPES
    assert CALENDAR_READONLY_SCOPE in SCOPES
    assert YOUTUBE_READONLY_SCOPE in SCOPES


def test_missing_scopes_detection():
    creds = MagicMock()
    creds.scopes = [GMAIL_READONLY_SCOPE]
    missing = missing_scopes(creds, SCOPES)
    assert CALENDAR_READONLY_SCOPE in missing
    assert YOUTUBE_READONLY_SCOPE in missing


def test_normalize_calendar_event_timed():
    normalized = normalize_calendar_event(
        {
            "id": "e1",
            "summary": "Interview",
            "status": "confirmed",
            "location": "Zoom",
            "description": "Bring resume",
            "start": {"dateTime": "2026-08-11T10:00:00+05:30"},
            "end": {"dateTime": "2026-08-11T11:00:00+05:30"},
        }
    )
    assert normalized is not None
    assert normalized["title"] == "Interview"
    assert normalized["all_day"] is False
    assert normalized["location"] == "Zoom"


def test_normalize_calendar_event_skips_cancelled():
    assert (
        normalize_calendar_event(
            {
                "id": "e2",
                "status": "cancelled",
                "start": {"dateTime": "2026-08-11T10:00:00+05:30"},
                "end": {"dateTime": "2026-08-11T11:00:00+05:30"},
            }
        )
        is None
    )


def test_normalize_calendar_all_day():
    normalized = normalize_calendar_event(
        {
            "id": "e3",
            "summary": "Holiday",
            "status": "confirmed",
            "start": {"date": "2026-08-11"},
            "end": {"date": "2026-08-12"},
        }
    )
    assert normalized is not None
    assert normalized["all_day"] is True


def test_calendar_service_get_today_events(settings_env):
    service = CalendarService(get_settings())
    mock_api = MagicMock()
    mock_api.events.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "id": "e1",
                "summary": "Standup",
                "status": "confirmed",
                "start": {"dateTime": "2026-08-11T09:00:00+05:30"},
                "end": {"dateTime": "2026-08-11T09:30:00+05:30"},
            }
        ]
    }
    with (
        patch("app.integrations.calendar.get_valid_credentials", return_value=MagicMock()),
        patch("app.integrations.calendar.build", return_value=mock_api),
    ):
        result = service.get_today_events()

    assert result["count"] == 1
    assert result["events"][0]["title"] == "Standup"
    assert "payload" not in result["events"][0]


def test_calendar_service_get_upcoming_events(settings_env):
    service = CalendarService(get_settings())
    mock_api = MagicMock()
    mock_api.events.return_value.list.return_value.execute.return_value = {"items": []}
    with (
        patch("app.integrations.calendar.get_valid_credentials", return_value=MagicMock()),
        patch("app.integrations.calendar.build", return_value=mock_api),
    ):
        result = service.get_upcoming_events(days=7)

    assert result["days"] == 7
    assert result["count"] == 0
    assert result["events"] == []


@pytest.mark.asyncio
async def test_calendar_mcp_tools(settings_env):
    mock_service = MagicMock()
    mock_service.get_today_events.return_value = {
        "date": "2026-08-11",
        "count": 1,
        "events": [
            {
                "id": "e1",
                "title": "Interview",
                "start": "2026-08-11T10:00:00+05:30",
                "end": "2026-08-11T11:00:00+05:30",
                "location": "",
                "description": "",
                "status": "confirmed",
                "all_day": False,
            }
        ],
    }
    mock_service.get_upcoming_events.return_value = {
        "from": "x",
        "to": "y",
        "days": 7,
        "count": 0,
        "events": [],
    }
    server = CalendarMCPServer(calendar_service=mock_service)
    names = [t.name for t in server.list_tools()]
    assert "calendar.get_today_events" in names
    assert "calendar.get_upcoming_events" in names

    today = await server.get_today_events()
    upcoming = await server.get_upcoming_events(days=7)
    assert today["success"] is True
    assert today["events"][0]["title"] == "Interview"
    assert upcoming["days"] == 7


def test_normalize_youtube_video():
    video = normalize_youtube_video(
        video_id="abc123",
        title="Demo",
        channel="My Channel",
        published_at="2026-08-11T12:00:00Z",
        description="Hello world",
    )
    assert video["url"] == "https://www.youtube.com/watch?v=abc123"
    assert video["channel"] == "My Channel"


def test_youtube_service_requires_channel_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("TIMEZONE", "Asia/Kolkata")
    monkeypatch.setenv("BRIEFING_TIME", "19:00")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("GOOGLE_TOKEN_PATH", str(tmp_path / "google_token.json"))
    monkeypatch.setenv("YOUTUBE_CHANNEL_IDS", "")
    get_settings.cache_clear()

    service = YouTubeService(get_settings())
    result = service.get_recent_videos()
    assert result["count"] == 0
    assert "YOUTUBE_CHANNEL_IDS" in result["message"]


def test_youtube_service_get_recent_videos(settings_env):
    service = YouTubeService(get_settings())
    mock_api = MagicMock()
    mock_api.channels.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "id": "UCaaaa",
                "contentDetails": {"relatedPlaylists": {"uploads": "UUaaaa"}},
                "snippet": {"title": "Channel A"},
            }
        ]
    }
    mock_api.playlistItems.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "snippet": {
                    "title": "New Video",
                    "channelTitle": "Channel A",
                    "description": "Desc",
                    "publishedAt": "2026-08-11T12:00:00Z",
                    "resourceId": {"videoId": "vid1"},
                },
                "contentDetails": {
                    "videoId": "vid1",
                    "videoPublishedAt": "2026-08-11T12:00:00Z",
                },
            }
        ]
    }
    with (
        patch("app.integrations.youtube.get_valid_credentials", return_value=MagicMock()),
        patch("app.integrations.youtube.build", return_value=mock_api),
    ):
        result = service.get_recent_videos(limit=5)

    assert result["count"] >= 1
    assert result["videos"][0]["video_id"] == "vid1"
    assert result["videos"][0]["url"].endswith("vid1")


@pytest.mark.asyncio
async def test_youtube_mcp_tool(settings_env):
    mock_service = MagicMock()
    mock_service.get_recent_videos.return_value = {
        "count": 1,
        "videos": [
            {
                "video_id": "vid1",
                "title": "Hello",
                "channel": "Chan",
                "published_at": "2026-08-11T12:00:00Z",
                "description": "x",
                "url": "https://www.youtube.com/watch?v=vid1",
            }
        ],
    }
    server = YouTubeMCPServer(youtube_service=mock_service)
    result = await server.get_recent_videos(limit=10)
    assert result["success"] is True
    assert result["videos"][0]["title"] == "Hello"


def test_default_mcp_client_registers_all_tools(settings_env):
    client = create_default_mcp_client()
    names = {t.name for t in client.list_tools()}
    assert names == {
        "gmail.get_today_emails",
        "calendar.get_today_events",
        "calendar.get_upcoming_events",
        "youtube.get_recent_videos",
    }


def test_plan_tools_for_calendar_and_overview():
    assert plan_tool_calls("What meetings do I have today?")[0].name == (
        "calendar.get_today_events"
    )
    assert detect_intent("What happened today?") is Intent.OVERVIEW_TODAY
    names = [c.name for c in plan_tool_calls("What happened today?")]
    assert "gmail.get_today_emails" in names
    assert "calendar.get_today_events" in names
    assert "youtube.get_recent_videos" in names


@pytest.mark.asyncio
async def test_agent_calendar_query(settings_env):
    mcp = MagicMock()
    mcp.list_tools.return_value = []
    mcp.call_tool = AsyncMock(
        return_value={
            "count": 1,
            "events": [
                {
                    "title": "Interview",
                    "start": "2026-08-11T10:00:00+05:30",
                    "location": "",
                }
            ],
        }
    )
    agent = PersonalAgent(mcp, LLMService(get_settings()))
    reply = await agent.process_message("What meetings do I have today?")
    mcp.call_tool.assert_awaited()
    assert "Interview" in reply
    assert "📅" in reply


@pytest.mark.asyncio
async def test_agent_youtube_query(settings_env):
    mcp = MagicMock()
    mcp.list_tools.return_value = []
    mcp.call_tool = AsyncMock(
        return_value={
            "count": 1,
            "videos": [
                {
                    "title": "Ship it",
                    "channel": "Favorite",
                    "url": "https://www.youtube.com/watch?v=1",
                }
            ],
        }
    )
    agent = PersonalAgent(mcp, LLMService(get_settings()))
    reply = await agent.process_message("Any new YouTube videos?")
    assert "Ship it" in reply
    assert "▶️" in reply


def test_formatters_calendar_youtube():
    assert "Interview" in format_calendar_events(
        {"count": 1, "events": [{"title": "Interview", "start": "10:00"}]}
    )
    assert "Favorite" in format_youtube_videos(
        {
            "count": 1,
            "videos": [
                {
                    "title": "V",
                    "channel": "Favorite",
                    "url": "https://www.youtube.com/watch?v=1",
                }
            ],
        }
    )


def test_evening_briefing_includes_sections():
    text = format_evening_briefing(
        gmail_data={"count": 2, "emails": [], "date": "2026-08-11"},
        calendar_data={
            "count": 1,
            "events": [{"title": "Interview", "start": "2026-08-12T10:00:00+05:30"}],
        },
        youtube_data={"count": 1, "videos": [{"title": "New", "channel": "Chan"}]},
        timezone="Asia/Kolkata",
    )
    assert "📧 Gmail" in text
    assert "📅 Calendar" in text
    assert "▶️ YouTube" in text
    assert "🔥 Priority" in text
    assert "Interview" in text
