"""LLM service abstraction (optional for Milestone 4)."""

from __future__ import annotations

import logging
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


class LLMService:
    """
    Thin chat-completion client.

    The agent never gives the LLM unrestricted tool access.
    Tool selection stays in the agent/router layer.
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
            return await self._openai_compatible(
                system_prompt=system_prompt,
                user_message=user_message,
                context=context,
            )
        if provider == "gemini":
            return await self._gemini(
                system_prompt=system_prompt,
                user_message=user_message,
                context=context,
            )
        raise LLMError(f"Unsupported LLM_PROVIDER '{provider}'")

    async def _openai_compatible(
        self,
        *,
        system_prompt: str,
        user_message: str,
        context: str | None,
    ) -> str:
        provider = self.settings.llm_provider
        base = (self.settings.llm_base_url or DEFAULT_BASE_URLS[provider]).rstrip("/")
        model = self.settings.llm_model or DEFAULT_MODELS[provider]
        messages: list[dict[str, str]] = [
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

        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }

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
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected LLM response shape") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM returned an empty response")
        return content.strip()

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
