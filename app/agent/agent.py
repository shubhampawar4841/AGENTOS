"""Conversational agent: allow-listed MCP tools → reply."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agent.formatters import (
    emails_context_for_llm,
    format_calendar_events,
    format_email_summary_deterministic,
    format_overview,
    format_today_emails,
    format_youtube_videos,
)
from app.agent.prompts import (
    COMBINED_REPLY_SYSTEM_PROMPT,
    GMAIL_SUMMARY_SYSTEM_PROMPT,
    START_MESSAGE,
    TOOL_SELECTION_SYSTEM_PROMPT,
    UNKNOWN_MESSAGE,
)
from app.agent.router import (
    ALLOWED_AGENT_TOOLS,
    Intent,
    ToolCall,
    detect_intent,
    plan_tool_calls,
)
from app.mcp import MCPError
from app.mcp.client import MCPClient
from app.services.llm import LLMError, LLMNotConfiguredError, LLMService

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Raised when the agent cannot complete a request."""


class PersonalAgent:
    """
    Telegram-facing agent.

    Tool selection is allow-listed. When an LLM is configured it may choose
    among those tools; otherwise a deterministic router is used.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        llm: LLMService | None = None,
    ) -> None:
        self._mcp = mcp_client
        self._llm = llm or LLMService()

    async def process_message(self, message: str) -> str:
        intent = detect_intent(message)
        logger.info("Agent intent=%s", intent.value)

        if intent is Intent.START:
            return START_MESSAGE

        tool_calls = await self._select_tools(message, intent)
        if not tool_calls:
            return UNKNOWN_MESSAGE

        results: dict[str, Any] = {}
        for call in tool_calls:
            results[call.name] = await self._call_allowed_tool(call.name, call.arguments)

        return await self._format_reply(message, intent, results)

    async def _select_tools(self, message: str, intent: Intent) -> list[ToolCall]:
        # Prefer deterministic plan for clear intents; use LLM for ambiguous multi-tool cases.
        planned = plan_tool_calls(message)
        if planned and intent is not Intent.OVERVIEW_TODAY and intent is not Intent.UNKNOWN:
            return planned

        if self._llm.enabled:
            llm_plan = await self._llm_select_tools(message)
            if llm_plan is not None:
                return llm_plan

        return planned

    async def _llm_select_tools(self, message: str) -> list[ToolCall] | None:
        catalog = []
        for tool in self._mcp.list_tools():
            if tool.name not in ALLOWED_AGENT_TOOLS:
                continue
            catalog.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
            )
        try:
            raw = await self._llm.generate_response(
                system_prompt=TOOL_SELECTION_SYSTEM_PROMPT,
                user_message=message,
                context=json.dumps({"allow_listed_tools": catalog}, ensure_ascii=True),
            )
        except (LLMNotConfiguredError, LLMError) as exc:
            logger.warning("LLM tool selection unavailable: %s", exc)
            return None

        parsed = _extract_json_object(raw)
        if not isinstance(parsed, dict):
            return None
        tools = parsed.get("tools")
        if not isinstance(tools, list):
            return None

        calls: list[ToolCall] = []
        for item in tools:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name not in ALLOWED_AGENT_TOOLS:
                logger.warning("LLM requested disallowed tool '%s'; ignored", name)
                continue
            args = item.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            calls.append(ToolCall(name=name, arguments=args))
        return calls

    async def _call_allowed_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if tool_name not in ALLOWED_AGENT_TOOLS:
            raise AgentError(f"Tool '{tool_name}' is not allowed for the agent")
        try:
            result = await self._mcp.call_tool(tool_name, arguments)
        except MCPError as exc:
            logger.error("Agent MCP tool failed: %s", exc)
            raise AgentError(str(exc)) from exc
        if not isinstance(result, dict):
            raise AgentError("MCP tool returned an unexpected payload")
        return result

    async def _format_reply(
        self,
        user_message: str,
        intent: Intent,
        results: dict[str, Any],
    ) -> str:
        if intent is Intent.GMAIL_SUMMARY and "gmail.get_today_emails" in results:
            return await self._summarize_emails(user_message, results["gmail.get_today_emails"])

        if len(results) == 1:
            name, data = next(iter(results.items()))
            if name == "gmail.get_today_emails":
                return format_today_emails(data)
            if name == "calendar.get_today_events":
                return format_calendar_events(data, heading="📅 Today's calendar")
            if name == "calendar.get_upcoming_events":
                return format_calendar_events(data, heading="📅 Upcoming calendar")
            if name == "youtube.get_recent_videos":
                return format_youtube_videos(data)

        overview = format_overview(
            gmail_data=results.get("gmail.get_today_emails"),
            calendar_data=results.get("calendar.get_today_events")
            or results.get("calendar.get_upcoming_events"),
            youtube_data=results.get("youtube.get_recent_videos"),
        )

        if self._llm.enabled and len(results) > 1:
            try:
                return await self._llm.generate_response(
                    system_prompt=COMBINED_REPLY_SYSTEM_PROMPT,
                    user_message=user_message,
                    context=json.dumps(results, ensure_ascii=True, default=str)[:8000],
                )
            except (LLMNotConfiguredError, LLMError) as exc:
                logger.warning("LLM combined reply unavailable: %s", exc)
        return overview

    async def _summarize_emails(self, user_message: str, data: dict[str, Any]) -> str:
        if self._llm.enabled:
            try:
                return await self._llm.generate_response(
                    system_prompt=GMAIL_SUMMARY_SYSTEM_PROMPT,
                    user_message=user_message,
                    context=emails_context_for_llm(data),
                )
            except (LLMNotConfiguredError, LLMError) as exc:
                logger.warning("LLM summary unavailable, using deterministic format: %s", exc)
        return format_email_summary_deterministic(data)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


async def process_message(message: str, agent: PersonalAgent) -> str:
    """Module-level helper used by the Telegram poller."""
    try:
        return await agent.process_message(message)
    except AgentError as exc:
        text = str(exc)
        lower = text.lower()
        if (
            "authenticate" in lower
            or "not connected" in lower
            or "revoked" in lower
            or "re-authenticate" in lower
            or ("missing" in lower and "scope" in lower)
        ):
            return (
                "🔐 Google access needs attention "
                "(missing scopes, expired token, or not connected).\n"
                "Open http://localhost:3000/auth/google to reconnect and grant "
                "Gmail + Calendar + YouTube read-only access."
            )
        return f"⚠️ I couldn't complete that request.\n{text}"
