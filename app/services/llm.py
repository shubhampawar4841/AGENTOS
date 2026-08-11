"""LLM service abstraction (optional for Milestone 4)."""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.1-8b-instant",
    "gemini": "gemini-1.5-flash",
}

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
}


class LLMError(Exception):
    """Raised when the LLM provider fails."""


class LLMNotConfiguredError(LLMError):
    """Raised when no LLM provider/API key is configured."""


@dataclass(frozen=True)
class LLMToolCall:
    """A structured tool request returned by a chat model."""

    id: str
    name: str
    arguments: dict[str, Any]
    argument_error: str | None = None


@dataclass(frozen=True)
class LLMChatResponse:
    """Normalized assistant message from a chat completion."""

    content: str | None
    tool_calls: tuple[LLMToolCall, ...] = ()


class LLMService:
    """
    Thin chat-completion client.

    The agent supplies an explicit allow-listed tool catalog. This service only
    handles provider transport and normalizes structured tool calls.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def enabled(self) -> bool:
        return self.settings.llm_configured

    async def generate_response(
        self,
        system_prompt: str,
        user_message: str,
        context: str | None = None,
    ) -> str:
        if not self.enabled:
            raise LLMNotConfiguredError(
                "LLM is not configured. Set LLM_PROVIDER and LLM_API_KEY."
            )

        provider = self.settings.llm_provider
        if provider in {"openai", "groq"}:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
            ]
            if context:
                messages.append(
                    {
                        "role": "system",
                        "content": f"Tool/context data (read-only):\n{context}",
                    }
                )
            messages.append({"role": "user", "content": user_message})
            response = await self.chat(messages)
            if not response.content:
                raise LLMError("LLM returned an empty response")
            return response.content
            )
        if provider == "gemini":
            return await self._gemini(
                system_prompt=system_prompt,
                user_message=user_message,
                context=context,
            )
        raise LLMError(f"Unsupported LLM_PROVIDER '{provider}'")

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMChatResponse:
        """Send a full conversation and optional native tools to the model."""
        if not self.enabled:
            raise LLMNotConfiguredError(
                "LLM is not configured. Set LLM_PROVIDER and LLM_API_KEY."
            )
        if self.settings.llm_provider not in {"openai", "groq"}:
            raise LLMError(
                "Conversational tool calling currently requires Groq or OpenAI"
            )
        return await self._openai_compatible(messages=messages, tools=tools)

    async def _openai_compatible(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMChatResponse:
        provider = self.settings.llm_provider
        base = (self.settings.llm_base_url or DEFAULT_BASE_URLS[provider]).rstrip("/")
        model = self.settings.llm_model or DEFAULT_MODELS[provider]

        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{base}/chat/completions",
                    headers=headers,
                    json=payload,
                )
        except httpx.HTTPError as exc:
            logger.error("LLM request failed: %s", type(exc).__name__)
            raise LLMError("Failed to reach LLM provider") from exc

        data = self._parse_json(response)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected LLM response shape") from exc

        content_value = message.get("content")
        content = content_value.strip() if isinstance(content_value, str) else None
        tool_calls = self._parse_tool_calls(message.get("tool_calls"))
        if not content and not tool_calls:
            raise LLMError("LLM returned an empty response")
        return LLMChatResponse(content=content, tool_calls=tuple(tool_calls))

    @staticmethod
    def _parse_tool_calls(raw_calls: Any) -> list[LLMToolCall]:
        if raw_calls is None:
            return []
        if not isinstance(raw_calls, list):
            raise LLMError("Unexpected LLM tool_calls shape")

        calls: list[LLMToolCall] = []
        for index, raw in enumerate(raw_calls):
            if not isinstance(raw, dict):
                raise LLMError("Unexpected LLM tool call shape")
            function = raw.get("function")
            if not isinstance(function, dict):
                raise LLMError("Unexpected LLM tool call function")
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise LLMError("LLM tool call is missing a name")

            raw_arguments = function.get("arguments", "{}")
            argument_error: str | None = None
            arguments: dict[str, Any] = {}
            if isinstance(raw_arguments, str):
                try:
                    parsed = json.loads(raw_arguments)
                    if isinstance(parsed, dict):
                        arguments = parsed
                    else:
                        argument_error = "Tool arguments must be a JSON object"
                except json.JSONDecodeError:
                    argument_error = "Tool arguments were not valid JSON"
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                argument_error = "Tool arguments must be a JSON object"

            call_id = raw.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = f"tool_call_{index}"
            calls.append(
                LLMToolCall(
                    id=call_id,
                    name=name.strip(),
                    arguments=arguments,
                    argument_error=argument_error,
                )
            )
        return calls

    async def _gemini(
        self,
        *,
        system_prompt: str,
        user_message: str,
        context: str | None,
    ) -> str:
        model = self.settings.llm_model or DEFAULT_MODELS["gemini"]
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        parts = [system_prompt]
        if context:
            parts.append(f"Tool/context data (read-only):\n{context}")
        parts.append(f"User:\n{user_message}")
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": "\n\n".join(parts)}]}],
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url,
                    params={"key": self.settings.llm_api_key},
                    json=payload,
                )
        except httpx.HTTPError as exc:
            logger.error("Gemini request failed: %s", type(exc).__name__)
            raise LLMError("Failed to reach Gemini") from exc

        data = self._parse_json(response)
        try:
            content = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected Gemini response shape") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Gemini returned an empty response")
        return content.strip()

    @staticmethod
    def _parse_json(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            logger.error("LLM returned non-JSON (status=%s)", response.status_code)
            raise LLMError("Invalid LLM response") from exc
        if response.status_code >= 400:
            # Never log API keys; provider error bodies are usually safe.
            logger.error("LLM API error (status=%s)", response.status_code)
            raise LLMError(f"LLM API error (status={response.status_code})")
        if not isinstance(data, dict):
            raise LLMError("Invalid LLM response")
        return data
