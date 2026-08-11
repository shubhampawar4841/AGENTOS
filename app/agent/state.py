"""Bounded, process-local conversation memory for SYNCOS."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

ConversationHandler = Callable[[str, list[dict[str, str]]], Awaitable[str]]


class ConversationStore:
    """Keep a small, isolated message history for each Telegram chat."""

    def __init__(self, max_messages: int = 20) -> None:
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        self.max_messages = max_messages
        self._histories: dict[str, deque[dict[str, str]]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _history(self, chat_id: str) -> deque[dict[str, str]]:
        key = str(chat_id)
        if key not in self._histories:
            self._histories[key] = deque(maxlen=self.max_messages)
        return self._histories[key]

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
        """Atomically read history, process a turn, and remember its outcome."""
        async with self._lock(chat_id):
            history = [dict(item) for item in self._history(chat_id)]
            reply = await handler(message, history)
            conversation = self._history(chat_id)
            conversation.append({"role": "user", "content": message})
            conversation.append({"role": "assistant", "content": reply})
            return reply

    async def get_history(self, chat_id: str) -> list[dict[str, str]]:
        async with self._lock(chat_id):
            return [dict(item) for item in self._history(chat_id)]

    async def clear(self, chat_id: str) -> None:
        async with self._lock(chat_id):
            self._histories.pop(str(chat_id), None)

    @property
    def conversation_count(self) -> int:
        return len(self._histories)
