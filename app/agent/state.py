"""Bounded, process-local conversation memory and agent run state for SYNCOS."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolExecution:
    """
    Record of one real MCP tool execution.

    Only the MCP executor may create these; model output never counts as an
    execution.
    """

    tool_name: str
    success: bool
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class AgentRunResult:
    """Reply plus the provenance of every tool executed during the run."""

    response: str
    tools_executed: tuple[ToolExecution, ...] = field(default_factory=tuple)

    @property
    def executed_tool_names(self) -> tuple[str, ...]:
        return tuple(execution.tool_name for execution in self.tools_executed)

    @property
    def successful_tool_names(self) -> tuple[str, ...]:
        return tuple(
            execution.tool_name for execution in self.tools_executed if execution.success
        )

    def verified_results(self) -> dict[str, Any]:
        """Tool results that actually came back successfully from MCP."""
        return {
            execution.tool_name: execution.result
            for execution in self.tools_executed
            if execution.success and execution.result is not None
        }


ConversationHandler = Callable[
    [str, list[dict[str, str]], dict[str, Any]],
    Awaitable["AgentRunResult | str"],
]


class ConversationStore:
    """
    Keep a small, isolated message history for each Telegram chat.

    Verified tool results are stored separately from chat text so a previous
    (possibly hallucinated) assistant message can never be treated as evidence.
    """

    def __init__(self, max_messages: int = 20) -> None:
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        self.max_messages = max_messages
        self._histories: dict[str, deque[dict[str, str]]] = {}
        self._verified: dict[str, dict[str, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _history(self, chat_id: str) -> deque[dict[str, str]]:
        key = str(chat_id)
        if key not in self._histories:
            self._histories[key] = deque(maxlen=self.max_messages)
        return self._histories[key]

    def _verified_results(self, chat_id: str) -> dict[str, Any]:
        key = str(chat_id)
        if key not in self._verified:
            self._verified[key] = {}
        return self._verified[key]

    def _lock(self, chat_id: str) -> asyncio.Lock:
        key = str(chat_id)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def process_turn(
        self,
        chat_id: str,
        message: str,
        handler: ConversationHandler,
    ) -> str:
        """Atomically read state, process a turn, and remember its outcome."""
        async with self._lock(chat_id):
            history = [dict(item) for item in self._history(chat_id)]
            verified = dict(self._verified_results(chat_id))

            outcome = await handler(message, history, verified)
            if isinstance(outcome, AgentRunResult):
                reply = outcome.response
                self._verified_results(chat_id).update(outcome.verified_results())
            else:
                reply = str(outcome)

            conversation = self._history(chat_id)
            conversation.append({"role": "user", "content": message})
            conversation.append({"role": "assistant", "content": reply})
            return reply

    async def get_history(self, chat_id: str) -> list[dict[str, str]]:
        async with self._lock(chat_id):
            return [dict(item) for item in self._history(chat_id)]

    async def get_verified_results(self, chat_id: str) -> dict[str, Any]:
        async with self._lock(chat_id):
            return dict(self._verified_results(chat_id))

    async def clear(self, chat_id: str) -> None:
        async with self._lock(chat_id):
            self._histories.pop(str(chat_id), None)
            self._verified.pop(str(chat_id), None)

    @property
    def conversation_count(self) -> int:
        return len(self._histories)
