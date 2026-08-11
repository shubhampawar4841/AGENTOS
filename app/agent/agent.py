"""SYNCOS conversational agent with native LLM tool calling and provenance checks."""

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
from app.agent.guardrails import (
    review_assistant_content,
    strip_raw_tool_syntax,
    tool_service,
)
from app.agent.prompts import (
    START_MESSAGE,
    SYNCOS_SYSTEM_PROMPT,
    TOOL_ENFORCEMENT_REMINDER,
    UNKNOWN_MESSAGE,
)
from app.agent.router import (
    ALLOWED_AGENT_TOOLS,
    Intent,
    detect_intent,
    plan_tool_calls,
)
from app.agent.state import AgentRunResult, ToolExecution
from app.mcp import MCPError
from app.mcp.client import MCPClient
from app.services.llm import LLMChatResponse, LLMError, LLMService, LLMToolCall

logger = logging.getLogger(__name__)

LLM_TROUBLE_MESSAGE = (
    "I'm having trouble thinking right now. Please try again in a moment."
)


class AgentError(Exception):
    """Raised when the agent cannot complete a request."""


class PersonalAgent:
    """Telegram-facing, history-aware agent backed by Groq and MCP."""

    MAX_TOOL_ROUNDS = 4
    MAX_TOOL_RESULT_CHARS = 12_000
    MAX_VERIFIED_CONTEXT_CHARS = 6_000

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
        verified_context: dict[str, Any] | None = None,
    ) -> str:
        """Process one turn and return only the user-facing reply."""
        result = await self.run(message, conversation_history, verified_context)
        return result.response

    async def run(
        self,
        message: str,
        conversation_history: list[dict[str, Any]] | None = None,
        verified_context: dict[str, Any] | None = None,
    ) -> AgentRunResult:
        """
        Process one user turn using recent history and a verified MCP tool loop.

        External data may only be described when a tool execution in this run,
        or a verified result carried in `verified_context`, actually returned it.
        """
        text = (message or "").strip()
        if not text:
            return AgentRunResult(response="What would you like help with?")

        logger.info("Agent request received (chars=%d)", len(text))
        if not self._llm.enabled:
            return await self._deterministic_fallback(text)

        tool_log: list[ToolExecution] = []
        verified_context = verified_context or {}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYNCOS_SYSTEM_PROMPT}
        ]
        context_message = self._verified_context_message(verified_context)
        if context_message:
            messages.append(context_message)
        messages.extend(self._clean_history(conversation_history or []))
        messages.append({"role": "user", "content": text})

        tools = self._tool_catalog()
        enforcement_used = False

        try:
            for _round in range(self.MAX_TOOL_ROUNDS):
                response = await self._llm.chat(messages, tools=tools)

                if response.tool_calls:
                    messages.append(self._assistant_tool_message(response))
                    for call in response.tool_calls:
                        logger.info("LLM requested tool: %s", call.name)
                        payload, execution = await self._execute_tool_call(call)
                        tool_log.append(execution)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "name": call.name,
                                "content": payload,
                            }
                        )
                        logger.info("Returning tool result to LLM: %s", execution.tool_name)
                    continue

                verdict = review_assistant_content(
                    response.content or "",
                    self._verified_services(tool_log, verified_context),
                )
                if verdict.ok:
                    logger.info("LLM final response generated")
                    return AgentRunResult(
                        response=verdict.safe_response or "Could you rephrase that?",
                        tools_executed=tuple(tool_log),
                    )

                if not enforcement_used:
                    enforcement_used = True
                    logger.warning(
                        "Blocked unverified response (reason=%s, services=%s); "
                        "requiring real tool execution",
                        verdict.reason,
                        ",".join(verdict.unverified_services) or "none",
                    )
                    messages.append(
                        {"role": "system", "content": TOOL_ENFORCEMENT_REMINDER}
                    )
                    continue

                logger.error(
                    "Refusing to send unverified response (reason=%s)", verdict.reason
                )
                return AgentRunResult(
                    response=verdict.safe_response,
                    tools_executed=tuple(tool_log),
                )
        except LLMError as exc:
            logger.error("SYNCOS LLM failed: %s", exc)
            return AgentRunResult(
                response=LLM_TROUBLE_MESSAGE,
                tools_executed=tuple(tool_log),
            )

        logger.warning("SYNCOS reached the tool-call round limit")
        return AgentRunResult(
            response="I couldn't finish that safely. Please narrow the request and try again.",
            tools_executed=tuple(tool_log),
        )

    @staticmethod
    def _verified_services(
        tool_log: list[ToolExecution],
        verified_context: dict[str, Any],
    ) -> set[str]:
        """Services with real data available, from this run or verified context."""
        services: set[str] = set()
        for execution in tool_log:
            if execution.success:
                service = tool_service(execution.tool_name)
                if service:
                    services.add(service)
        for tool_name in verified_context:
            service = tool_service(tool_name)
            if service:
                services.add(service)
        return services

    def _verified_context_message(
        self,
        verified_context: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not verified_context:
            return None
        payload = json.dumps(verified_context, ensure_ascii=False, default=str)[
            : self.MAX_VERIFIED_CONTEXT_CHARS
        ]
        return {
            "role": "system",
            "content": (
                "Verified tool results from earlier in this conversation. This data "
                "came from real tool executions and is safe to reference. Anything "
                "not present here has NOT been retrieved:\n" + payload
            ),
        }

    def _tool_catalog(self) -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for tool in self._mcp.list_tools():
            if tool.name in ALLOWED_AGENT_TOOLS:
                catalog.append(
                    {
                        "type": "function",
                        "function": {
                            # OpenAI-compatible APIs restrict function names to
                            # letters, numbers, underscores, and hyphens.
                            "name": self._provider_tool_name(tool.name),
                            "description": tool.description,
                            "parameters": tool.input_schema
                            or {"type": "object", "properties": {}},
                        },
                    }
                )
        return catalog

    @staticmethod
    def _provider_tool_name(tool_name: str) -> str:
        return tool_name.replace(".", "__")

    @classmethod
    def _resolve_tool_name(cls, provider_name: str) -> str | None:
        if provider_name in ALLOWED_AGENT_TOOLS:
            return provider_name
        for name in ALLOWED_AGENT_TOOLS:
            if cls._provider_tool_name(name) == provider_name:
                return name
        return None

    @staticmethod
    def _clean_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clean: list[dict[str, Any]] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content:
                clean.append({"role": role, "content": strip_raw_tool_syntax(content)})
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

    async def _execute_tool_call(
        self,
        call: LLMToolCall,
    ) -> tuple[str, ToolExecution]:
        """Execute one structured tool call and record verifiable provenance."""
        tool_name = self._resolve_tool_name(call.name)
        if tool_name is None:
            logger.warning("Model requested disallowed tool '%s'", call.name)
            return self._failed_execution(
                call.name,
                "That capability is not available to this assistant.",
            )
        if call.argument_error:
            logger.warning("Malformed tool arguments for %s", tool_name)
            return self._failed_execution(tool_name, call.argument_error)

        try:
            tool = self._mcp.get_tool(tool_name)
            validation_error = _validate_arguments(call.arguments, tool.input_schema)
        except MCPError:
            validation_error = "The requested tool is not available."
        if validation_error:
            logger.warning("Rejected tool arguments for %s", tool_name)
            return self._failed_execution(tool_name, validation_error)

        logger.info("Executing MCP tool: %s", tool_name)
        try:
            result = await self._mcp.call_tool(tool_name, call.arguments)
        except MCPError as exc:
            logger.error("MCP tool failed: %s: %s", tool_name, exc)
            return self._failed_execution(tool_name, str(exc))
        if not isinstance(result, dict):
            logger.error("MCP tool returned unexpected payload: %s", tool_name)
            return self._failed_execution(
                tool_name,
                "The connected service returned an unexpected response.",
            )

        logger.info("MCP tool success: %s", tool_name)
        payload = json.dumps(result, ensure_ascii=False, default=str)[
            : self.MAX_TOOL_RESULT_CHARS
        ]
        return payload, ToolExecution(tool_name=tool_name, success=True, result=result)

    @staticmethod
    def _failed_execution(tool_name: str, error: str) -> tuple[str, ToolExecution]:
        payload = json.dumps({"success": False, "error": error}, ensure_ascii=False)
        return payload, ToolExecution(tool_name=tool_name, success=False, error=error)

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
            logger.error("MCP tool failed: %s: %s", tool_name, exc)
            raise AgentError(str(exc)) from exc
        if not isinstance(result, dict):
            raise AgentError("MCP tool returned an unexpected payload")
        return result

    async def _deterministic_fallback(self, message: str) -> AgentRunResult:
        """Compatibility path for installations with no LLM configured."""
        intent = detect_intent(message)
        if intent is Intent.START:
            return AgentRunResult(response=START_MESSAGE)
        calls = plan_tool_calls(message)
        if not calls:
            return AgentRunResult(response=UNKNOWN_MESSAGE)

        results: dict[str, dict[str, Any]] = {}
        executions: list[ToolExecution] = []
        for call in calls:
            data = await self._call_allowed_tool(call.name, call.arguments)
            results[call.name] = data
            executions.append(
                ToolExecution(tool_name=call.name, success=True, result=data)
            )

        reply = self._format_deterministic(intent, results)
        return AgentRunResult(response=reply, tools_executed=tuple(executions))

    @staticmethod
    def _format_deterministic(
        intent: Intent,
        results: dict[str, dict[str, Any]],
    ) -> str:
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


def _auth_error_reply(text: str) -> str | None:
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
    return None


async def run_agent_turn(
    message: str,
    agent: PersonalAgent,
    conversation_history: list[dict[str, Any]] | None = None,
    verified_context: dict[str, Any] | None = None,
) -> AgentRunResult:
    """Run one turn and return the reply plus tool provenance."""
    try:
        return await agent.run(message, conversation_history, verified_context)
    except AgentError as exc:
        text = str(exc)
        return AgentRunResult(response=_auth_error_reply(text) or f"⚠️ I couldn't complete that request.\n{text}")


async def process_message(
    message: str,
    agent: PersonalAgent,
    conversation_history: list[dict[str, Any]] | None = None,
    verified_context: dict[str, Any] | None = None,
) -> str:
    """Module-level helper returning only the user-facing reply."""
    result = await run_agent_turn(
        message,
        agent,
        conversation_history,
        verified_context,
    )
    return result.response
