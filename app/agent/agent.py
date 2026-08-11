"""SYNCOS conversational agent with native LLM tool calling."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.agent.formatters import (
    format_calendar_events,
    format_email_summary_deterministic,
    format_overview,
    format_today_emails,
    format_youtube_videos,
)
from app.agent.prompts import START_MESSAGE, SYNCOS_SYSTEM_PROMPT, UNKNOWN_MESSAGE
from app.agent.router import (
    ALLOWED_AGENT_TOOLS,
    Intent,
    detect_intent,
    plan_tool_calls,
)
from app.mcp import MCPError
from app.mcp.client import MCPClient
from app.services.llm import LLMChatResponse, LLMError, LLMService, LLMToolCall

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Raised when the agent cannot complete a request."""


class PersonalAgent:
    """Telegram-facing, history-aware agent backed by Groq and MCP."""

    MAX_TOOL_ROUNDS = 4
    MAX_TOOL_RESULT_CHARS = 12_000

    def __init__(
        self,
        mcp_client: MCPClient,
        llm: LLMService | None = None,
    ) -> None:
        self._mcp = mcp_client
        self._llm = llm or LLMService()

    async def process_message(
        self,
        message: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Process one user turn using recent history and an MCP tool loop."""
        text = (message or "").strip()
        if not text:
            return "What would you like help with?"
        if not self._llm.enabled:
            return await self._deterministic_fallback(text)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYNCOS_SYSTEM_PROMPT},
            *self._clean_history(conversation_history or []),
            {"role": "user", "content": text},
        ]
        tools = self._tool_catalog()

        try:
            for _round in range(self.MAX_TOOL_ROUNDS):
                response = await self._llm.chat(messages, tools=tools)
                if not response.tool_calls:
                    return response.content or "Could you rephrase that?"

                messages.append(self._assistant_tool_message(response))
                for call in response.tool_calls:
                    result = await self._execute_tool_call(call)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": result,
                        }
                    )
        except LLMError as exc:
            logger.error("SYNCOS LLM failed: %s", exc)
            return "I'm having trouble thinking right now. Please try again in a moment."

        logger.warning("SYNCOS reached the tool-call round limit")
        return "I couldn't finish that safely. Please narrow the request and try again."

    def _tool_catalog(self) -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for tool in self._mcp.list_tools():
            if tool.name in ALLOWED_AGENT_TOOLS:
                catalog.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema
                            or {"type": "object", "properties": {}},
                        },
                    }
                )
        return catalog

    @staticmethod
    def _clean_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clean: list[dict[str, Any]] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content:
                clean.append({"role": role, "content": content})
        return clean

    @staticmethod
    def _assistant_tool_message(response: LLMChatResponse) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": response.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=True),
                    },
                }
                for call in response.tool_calls
            ],
        }

    async def _execute_tool_call(self, call: LLMToolCall) -> str:
        if call.name not in ALLOWED_AGENT_TOOLS:
            logger.warning("Model requested disallowed tool '%s'", call.name)
            return self._tool_result_json(
                success=False,
                error="That capability is not available to this assistant.",
            )
        if call.argument_error:
            return self._tool_result_json(success=False, error=call.argument_error)

        try:
            tool = self._mcp.get_tool(call.name)
            validation_error = _validate_arguments(call.arguments, tool.input_schema)
        except MCPError:
            validation_error = "The requested tool is not available."
        if validation_error:
            return self._tool_result_json(success=False, error=validation_error)

        try:
            result = await self._mcp.call_tool(call.name, call.arguments)
        except MCPError as exc:
            logger.error("Agent MCP tool failed (%s): %s", call.name, exc)
            return self._tool_result_json(success=False, error=str(exc))
        if not isinstance(result, dict):
            return self._tool_result_json(
                success=False,
                error="The connected service returned an unexpected response.",
            )
        return json.dumps(result, ensure_ascii=False, default=str)[
            : self.MAX_TOOL_RESULT_CHARS
        ]

    @staticmethod
    def _tool_result_json(*, success: bool, error: str) -> str:
        return json.dumps({"success": success, "error": error}, ensure_ascii=False)

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

    async def _deterministic_fallback(self, message: str) -> str:
        """Compatibility path for tests or installations with no LLM configured."""
        intent = detect_intent(message)
        if intent is Intent.START:
            return START_MESSAGE
        calls = plan_tool_calls(message)
        if not calls:
            return UNKNOWN_MESSAGE
        results: dict[str, dict[str, Any]] = {}
        for call in calls:
            results[call.name] = await self._call_allowed_tool(call.name, call.arguments)
        if len(results) == 1:
            name, data = next(iter(results.items()))
            if name == "gmail.get_today_emails":
                if intent is Intent.GMAIL_SUMMARY:
                    return format_email_summary_deterministic(data)
                return format_today_emails(data)
            if name == "calendar.get_today_events":
                return format_calendar_events(data, heading="📅 Today's calendar")
            if name == "calendar.get_upcoming_events":
                return format_calendar_events(data, heading="📅 Upcoming calendar")
            if name == "youtube.get_recent_videos":
                return format_youtube_videos(data)
        return format_overview(
            gmail_data=results.get("gmail.get_today_emails"),
            calendar_data=results.get("calendar.get_today_events")
            or results.get("calendar.get_upcoming_events"),
            youtube_data=results.get("youtube.get_recent_videos"),
        )


def _validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> str | None:
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    for name in required:
        if name not in arguments:
            return f"Missing required argument: {name}"
    for name, value in arguments.items():
        rule = properties.get(name)
        if not isinstance(rule, dict):
            return f"Unknown argument: {name}"
        expected = rule.get("type")
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return f"Argument '{name}' must be an integer"
        if expected == "string" and not isinstance(value, str):
            return f"Argument '{name}' must be a string"
        minimum = rule.get("minimum")
        maximum = rule.get("maximum")
        if isinstance(value, (int, float)):
            if isinstance(minimum, (int, float)) and value < minimum:
                return f"Argument '{name}' must be at least {minimum}"
            if isinstance(maximum, (int, float)) and value > maximum:
                return f"Argument '{name}' must be at most {maximum}"
    return None


async def process_message(
    message: str,
    agent: PersonalAgent,
    conversation_history: list[dict[str, Any]] | None = None,
) -> str:
    """Module-level helper used by the Telegram poller."""
    try:
        return await agent.process_message(message, conversation_history)
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
