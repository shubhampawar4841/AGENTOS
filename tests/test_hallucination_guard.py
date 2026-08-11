"""Hallucination-prevention tests: no external-data claim without verified MCP data."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.agent import PersonalAgent
from app.agent.guardrails import (
    claims_external_data,
    contains_raw_tool_syntax,
    review_assistant_content,
    strip_raw_tool_syntax,
)
from app.agent.state import ConversationStore
from app.config import get_settings
from app.mcp import MCPError, MCPTool
from app.mcp.client import MCPClient
from app.services.llm import LLMChatResponse, LLMService, LLMToolCall

GMAIL_TOOL = "gmail.get_today_emails"
GMAIL_PROVIDER_TOOL = "gmail__get_today_emails"

FAKE_EMAILS = {
    "success": True,
    "count": 5,
    "emails": [
        {"sender": "QuantumLoopAI", "subject": "Interview invitation"},
        {"sender": "GitHub", "subject": "PR review requested"},
        {"sender": "Bank", "subject": "Statement ready"},
        {"sender": "Newsletter", "subject": "Weekly digest"},
        {"sender": "Friend", "subject": "Dinner?"},
    ],
}


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _gmail_tool() -> MCPTool:
    return MCPTool(
        name=GMAIL_TOOL,
        description="Read today's emails",
        handler=AsyncMock(),
        input_schema={"type": "object", "properties": {}},
    )


def _agent(responses: list[LLMChatResponse]) -> tuple[PersonalAgent, MagicMock, MagicMock]:
    llm = MagicMock(spec=LLMService)
    llm.enabled = True
    llm.chat = AsyncMock(side_effect=responses)
    mcp = MagicMock(spec=MCPClient)
    tools = [_gmail_tool()]
    mcp.list_tools.return_value = tools
    mcp.get_tool.side_effect = lambda name: next(t for t in tools if t.name == name)
    mcp.call_tool = AsyncMock()
    return PersonalAgent(mcp, llm), mcp, llm


def _tool_call_response() -> LLMChatResponse:
    return LLMChatResponse(
        content=None,
        tool_calls=(LLMToolCall("call-1", GMAIL_PROVIDER_TOOL, {}),),
    )


# --- Test A: tool succeeds -> verified provenance, real summary allowed --------


@pytest.mark.asyncio
async def test_a_gmail_success_marks_execution_and_allows_summary():
    agent, mcp, _llm = _agent(
        [
            _tool_call_response(),
            LLMChatResponse(
                content=(
                    "📧 I checked your inbox — you have 5 emails today. "
                    "Top 3: QuantumLoopAI interview, GitHub PR review, Bank statement."
                )
            ),
        ]
    )
    mcp.call_tool.return_value = FAKE_EMAILS

    result = await agent.run("give me top 3 important emails")

    mcp.call_tool.assert_awaited_once_with(GMAIL_TOOL, {})
    assert result.tools_executed[0].tool_name == GMAIL_TOOL
    assert result.tools_executed[0].success is True
    assert result.tools_executed[0].result == FAKE_EMAILS
    assert "QuantumLoopAI" in result.response
    assert result.verified_results() == {GMAIL_TOOL: FAKE_EMAILS}


# --- Test B: tool fails -> no fabricated email content ------------------------


@pytest.mark.asyncio
async def test_b_gmail_failure_never_fabricates_emails():
    agent, mcp, _llm = _agent(
        [
            _tool_call_response(),
            LLMChatResponse(content="I checked your inbox and you have 5 emails today."),
            LLMChatResponse(content="I couldn't access your Gmail. Please reconnect Google."),
        ]
    )
    mcp.call_tool.side_effect = MCPError("Google account is not connected")

    result = await agent.run("what are my top 3 important emails?")

    assert result.tools_executed[0].success is False
    assert "5 emails" not in result.response
    lowered = result.response.lower()
    assert "couldn't access" in lowered or "could not access" in lowered
    assert "gmail" in lowered


@pytest.mark.asyncio
async def test_b_persistent_fabrication_is_replaced_with_safe_message():
    agent, mcp, _llm = _agent(
        [
            _tool_call_response(),
            LLMChatResponse(content="I checked your inbox: you have 5 emails today."),
            LLMChatResponse(content="You received 5 emails, two look important."),
        ]
    )
    mcp.call_tool.side_effect = MCPError("Google token revoked")

    result = await agent.run("summarize my emails")

    assert result.response == (
        "I couldn't access your Gmail right now. "
        "Please reconnect your Google account and try again."
    )


# --- Test C: model claims execution without requesting a tool -----------------


@pytest.mark.asyncio
async def test_c_claim_without_tool_call_forces_real_execution():
    agent, mcp, llm = _agent(
        [
            LLMChatResponse(content="I've checked your Gmail and found 5 emails."),
            _tool_call_response(),
            LLMChatResponse(content="You have 5 emails today, including a GitHub PR review."),
        ]
    )
    mcp.call_tool.return_value = FAKE_EMAILS

    result = await agent.run("tell me my most important mails")

    mcp.call_tool.assert_awaited_once_with(GMAIL_TOOL, {})
    assert result.tools_executed[0].success is True
    assert llm.chat.await_count == 3
    assert "GitHub" in result.response


@pytest.mark.asyncio
async def test_c_claim_without_tools_and_no_recovery_is_not_returned():
    agent, mcp, _llm = _agent(
        [
            LLMChatResponse(content="I've checked your Gmail and found 5 emails."),
            LLMChatResponse(content="It looks like you have 5 emails from today."),
        ]
    )

    result = await agent.run("tell me my most important mails")

    mcp.call_tool.assert_not_awaited()
    assert result.tools_executed == ()
    assert "found 5 emails" not in result.response
    assert "couldn't access your Gmail" in result.response


# --- Test D: raw function syntax in content -----------------------------------


@pytest.mark.asyncio
async def test_d_raw_function_syntax_never_reaches_user():
    agent, mcp, _llm = _agent(
        [
            LLMChatResponse(content="<function=gmail__get_today_emails>{}</function>"),
            LLMChatResponse(content="<function=gmail__get_today_emails>{}</function>"),
        ]
    )

    result = await agent.run("check my emails")

    assert "<function" not in result.response
    assert "gmail__get_today_emails" not in result.response
    assert "checked" not in result.response.lower()
    assert "trouble accessing" in result.response.lower()
    mcp.call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_d_raw_function_syntax_recovers_via_native_tool_call():
    agent, mcp, _llm = _agent(
        [
            LLMChatResponse(content="<function=gmail__get_today_emails>{}</function>"),
            _tool_call_response(),
            LLMChatResponse(content="You have 5 emails today; the interview invite matters most."),
        ]
    )
    mcp.call_tool.return_value = FAKE_EMAILS

    result = await agent.run("check my emails")

    mcp.call_tool.assert_awaited_once_with(GMAIL_TOOL, {})
    assert "<function" not in result.response
    assert "interview invite" in result.response


# --- Test E: hallucinated history is not evidence -----------------------------


@pytest.mark.asyncio
async def test_e_hallucinated_history_is_not_treated_as_verified_data():
    agent, mcp, _llm = _agent(
        [
            LLMChatResponse(content="Those 5 emails were about work and billing."),
            _tool_call_response(),
            LLMChatResponse(content="Your 5 emails include an interview invite and a PR review."),
        ]
    )
    mcp.call_tool.return_value = FAKE_EMAILS
    history = [
        {"role": "user", "content": "check my mails"},
        {"role": "assistant", "content": "I found 5 emails today."},
    ]

    result = await agent.run("what were the emails about?", history)

    mcp.call_tool.assert_awaited_once_with(GMAIL_TOOL, {})
    assert result.tools_executed[0].success is True
    assert "interview invite" in result.response


@pytest.mark.asyncio
async def test_verified_context_allows_followup_without_new_tool_call():
    agent, mcp, _llm = _agent(
        [LLMChatResponse(content="You have 5 emails; the interview invite is most important.")]
    )

    result = await agent.run(
        "which one matters most?",
        [{"role": "user", "content": "check my mail"}],
        {GMAIL_TOOL: FAKE_EMAILS},
    )

    mcp.call_tool.assert_not_awaited()
    assert "interview invite" in result.response


# --- Casual conversation must stay unaffected ---------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        "Hey 👋 What's up?",
        "I can read your Gmail, Calendar, and YouTube uploads, but I can't send email.",
        "I can't send emails yet — Gmail access is read-only.",
        "Do you want today's schedule or the next 7 days?",
    ],
)
async def test_casual_and_honest_replies_are_not_blocked(reply):
    agent, mcp, _llm = _agent([LLMChatResponse(content=reply)])

    result = await agent.run("hey")

    assert result.response == reply
    mcp.call_tool.assert_not_awaited()


# --- Guardrail unit checks ----------------------------------------------------


def test_guardrail_detects_raw_syntax_and_strips_it():
    assert contains_raw_tool_syntax("<function=gmail__get_today_emails>{}</function>")
    assert contains_raw_tool_syntax("<|python_tag|>gmail__get_today_emails")
    assert not contains_raw_tool_syntax("I can read your Gmail.")
    assert strip_raw_tool_syntax("ok <function=x>{}</function>") == "ok"


def test_guardrail_claim_detection():
    assert claims_external_data("I checked your inbox and you have 5 emails")
    assert claims_external_data("Here are your latest videos")
    assert not claims_external_data("I couldn't access your Gmail right now")
    assert not claims_external_data("Do you want today's schedule or tomorrow's?")


def test_guardrail_allows_claims_when_service_verified():
    verdict = review_assistant_content(
        "I checked your inbox — you have 5 emails today.",
        {"gmail"},
    )
    assert verdict.ok is True

    blocked = review_assistant_content(
        "I checked your inbox — you have 5 emails today.",
        set(),
    )
    assert blocked.ok is False
    assert blocked.reason == "unverified_data_claim"
    assert blocked.unverified_services == ("gmail",)


@pytest.mark.asyncio
async def test_store_keeps_verified_results_separate_from_chat_text():
    from app.agent.state import AgentRunResult, ToolExecution

    store = ConversationStore(max_messages=6)

    async def handler(message, history, verified):
        assert verified == {}
        return AgentRunResult(
            response="You have 5 emails today.",
            tools_executed=(
                ToolExecution(tool_name=GMAIL_TOOL, success=True, result=FAKE_EMAILS),
            ),
        )

    await store.process_turn("chat-1", "check mail", handler)

    assert await store.get_verified_results("chat-1") == {GMAIL_TOOL: FAKE_EMAILS}
    assert await store.get_verified_results("chat-2") == {}
    history = await store.get_history("chat-1")
    assert [item["role"] for item in history] == ["user", "assistant"]
