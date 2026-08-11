"""Tests for SYNCOS native tool calling and short-term conversation memory."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.agent.agent import PersonalAgent
from app.agent.state import ConversationStore
from app.config import get_settings
from app.mcp import MCPError, MCPTool
from app.mcp.client import MCPClient
from app.services.llm import (
    LLMChatResponse,
    LLMError,
    LLMService,
    LLMToolCall,
)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def groq_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("TIMEZONE", "Asia/Kolkata")
    monkeypatch.setenv("BRIEFING_TIME", "19:00")
    monkeypatch.setenv("GOOGLE_TOKEN_PATH", str(tmp_path / "token.json"))
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_API_KEY", "test-groq-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://groq.test/v1")
    get_settings.cache_clear()
    return get_settings()


def _tool(name: str, properties: dict | None = None) -> MCPTool:
    return MCPTool(
        name=name,
        description=f"Read data using {name}",
        handler=AsyncMock(),
        input_schema={"type": "object", "properties": properties or {}},
    )


def _mock_agent(
    responses: list[LLMChatResponse],
    tools: list[MCPTool] | None = None,
) -> tuple[PersonalAgent, MagicMock, MagicMock]:
    llm = MagicMock(spec=LLMService)
    llm.enabled = True
    llm.chat = AsyncMock(side_effect=responses)
    mcp = MagicMock(spec=MCPClient)
    registered = tools or []
    mcp.list_tools.return_value = registered
    mcp.get_tool.side_effect = lambda name: next(t for t in registered if t.name == name)
    mcp.call_tool = AsyncMock()
    return PersonalAgent(mcp, llm), mcp, llm


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "reply"),
    [
        ("hey", "Hey 👋 What's up?"),
        ("now", "Do you mean right now, or today's updates?"),
        ("calendar", "Today's schedule, tomorrow, or the next 7 days?"),
        ("Send an email to John.", "I can't send email yet—Gmail is read-only."),
    ],
)
async def test_natural_messages_do_not_call_tools(message, reply):
    agent, mcp, _llm = _mock_agent([LLMChatResponse(content=reply)])

    assert await agent.process_message(message) == reply
    mcp.call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_tool_call_round_trip_uses_mcp_and_history():
    gmail = _tool("gmail.get_today_emails")
    agent, mcp, llm = _mock_agent(
        [
            LLMChatResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        id="call-1",
                        name="gmail__get_today_emails",
                        arguments={},
                    ),
                ),
            ),
            LLMChatResponse(content="📧 You received one important email from Ada."),
        ],
        [gmail],
    )
    mcp.call_tool.return_value = {
        "success": True,
        "count": 1,
        "emails": [{"sender": "Ada", "subject": "Interview"}],
    }
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hey!"},
    ]

    reply = await agent.process_message("Any important emails?", history)

    assert "Ada" in reply
    mcp.call_tool.assert_awaited_once_with("gmail.get_today_emails", {})
    second_messages = llm.chat.await_args_list[1].args[0]
    assert history[0] in second_messages
    assert any(item.get("role") == "tool" and "Interview" in item["content"] for item in second_messages)


@pytest.mark.asyncio
async def test_multiple_tools_can_be_called_in_one_round():
    gmail = _tool("gmail.get_today_emails")
    calendar = _tool(
        "calendar.get_upcoming_events",
        {"days": {"type": "integer", "minimum": 1, "maximum": 30}},
    )
    agent, mcp, _llm = _mock_agent(
        [
            LLMChatResponse(
                content=None,
                tool_calls=(
                    LLMToolCall("mail", "gmail__get_today_emails", {}),
                    LLMToolCall(
                        "cal",
                        "calendar__get_upcoming_events",
                        {"days": 2},
                    ),
                ),
            ),
            LLMChatResponse(content="🔥 Tomorrow's interview is the main priority."),
        ],
        [gmail, calendar],
    )
    mcp.call_tool.side_effect = [
        {"count": 1, "emails": [{"subject": "Interview prep"}]},
        {"count": 1, "events": [{"title": "Interview"}]},
    ]

    reply = await agent.process_message("What's important tomorrow?")

    assert "interview" in reply.lower()
    assert mcp.call_tool.await_count == 2


@pytest.mark.asyncio
async def test_invalid_or_disallowed_tool_calls_never_reach_mcp():
    calendar = _tool(
        "calendar.get_upcoming_events",
        {"days": {"type": "integer", "minimum": 1, "maximum": 30}},
    )
    agent, mcp, llm = _mock_agent(
        [
            LLMChatResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        "bad-args",
                        "calendar__get_upcoming_events",
                        {"days": 99},
                    ),
                    LLMToolCall("write", "gmail__send_email", {}),
                ),
            ),
            LLMChatResponse(content="I can't complete that request."),
        ],
        [calendar],
    )

    await agent.process_message("Do something unsafe")

    mcp.call_tool.assert_not_awaited()
    tool_messages = [
        item
        for item in llm.chat.await_args_list[1].args[0]
        if item.get("role") == "tool"
    ]
    assert len(tool_messages) == 2
    assert any("at most 30" in item["content"] for item in tool_messages)
    assert any("not available" in item["content"] for item in tool_messages)


@pytest.mark.asyncio
async def test_mcp_error_is_returned_to_model_for_natural_explanation():
    youtube = _tool("youtube.get_recent_videos")
    agent, mcp, _llm = _mock_agent(
        [
            LLMChatResponse(
                content=None,
                tool_calls=(
                    LLMToolCall("yt", "youtube__get_recent_videos", {}),
                ),
            ),
            LLMChatResponse(
                content="I couldn't access YouTube. Please reconnect your Google account."
            ),
        ],
        [youtube],
    )
    mcp.call_tool.side_effect = MCPError("Google token revoked")

    reply = await agent.process_message("Any new videos?")

    assert "reconnect" in reply.lower()


@pytest.mark.asyncio
async def test_llm_failure_has_stable_user_facing_message():
    agent, _mcp, llm = _mock_agent([])
    llm.chat.side_effect = LLMError("provider unavailable")

    reply = await agent.process_message("hey")

    assert reply == "I'm having trouble thinking right now. Please try again in a moment."


@pytest.mark.asyncio
async def test_conversation_store_is_bounded_and_isolated():
    store = ConversationStore(max_messages=4)

    async def echo(
        message: str,
        history: list[dict[str, str]],
        verified: dict[str, object],
    ) -> str:
        return f"{len(history)}:{message}"

    await store.process_turn("chat-a", "one", echo)
    await store.process_turn("chat-a", "two", echo)
    await store.process_turn("chat-a", "three", echo)
    await store.process_turn("chat-b", "other", echo)

    history_a = await store.get_history("chat-a")
    history_b = await store.get_history("chat-b")
    assert len(history_a) == 4
    assert history_a[0]["content"] == "two"
    assert [item["content"] for item in history_b] == ["other", "0:other"]


@pytest.mark.asyncio
async def test_groq_chat_payload_and_tool_response_parsing(groq_settings):
    service = LLMService(groq_settings)
    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "youtube__get_recent_videos",
                                    "arguments": json.dumps({"limit": 5}),
                                },
                            }
                        ],
                    }
                }
            ]
        },
    )
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(return_value=client)
    context_manager.__aexit__ = AsyncMock(return_value=None)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "youtube__get_recent_videos",
                "description": "Recent uploads",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    with patch("app.services.llm.httpx.AsyncClient", return_value=context_manager):
        result = await service.chat(
            [{"role": "user", "content": "Any new videos?"}],
            tools=tools,
        )

    assert result.tool_calls[0].arguments == {"limit": 5}
    payload = client.post.await_args.kwargs["json"]
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"


def test_malformed_tool_arguments_are_normalized_as_error():
    response = LLMService._parse_tool_calls(
        [
            {
                "id": "bad",
                "function": {
                    "name": "calendar__get_upcoming_events",
                    "arguments": "{not-json",
                },
            }
        ]
    )
    assert response[0].arguments == {}
    assert response[0].argument_error == "Tool arguments were not valid JSON"
