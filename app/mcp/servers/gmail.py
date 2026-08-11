"""Gmail MCP server — read-only tools backed by the Gmail service."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.integrations.gmail import GmailError, GmailService
from app.integrations.google_auth import GoogleAuthError, GoogleAuthRequiredError
from app.mcp import MCPError, MCPServer, MCPTool

logger = logging.getLogger(__name__)


class GmailMCPServer(MCPServer):
    """
    MCP tool surface for Gmail.

    Pattern:
        MCP Tool → GmailService → Google Gmail API
    """

    name = "gmail"

    def __init__(self, gmail_service: GmailService | None = None) -> None:
        self._gmail_service = gmail_service

    def _service(self) -> GmailService:
        return self._gmail_service or GmailService()

    def list_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name="gmail.get_today_emails",
                description="Return today's emails from the connected Gmail account (read-only).",
                handler=self.get_today_emails,
                input_schema={"type": "object", "properties": {}},
            )
        ]

    async def get_today_emails(self) -> dict[str, Any]:
        """Call the Gmail service and return normalized structured data."""
        try:
            result = await asyncio.to_thread(self._service().get_today_emails)
        except GoogleAuthRequiredError as exc:
            logger.error("Gmail MCP tool requires authentication")
            raise MCPError(str(exc)) from exc
        except GoogleAuthError as exc:
            logger.error("Gmail MCP auth error: %s", exc)
            raise MCPError(str(exc)) from exc
        except GmailError as exc:
            logger.error("Gmail MCP API error: %s", exc)
            raise MCPError(str(exc)) from exc
        except Exception as exc:
            logger.error("Gmail MCP unexpected error: %s", type(exc).__name__)
            raise MCPError("gmail.get_today_emails failed unexpectedly") from exc

        emails = result.get("emails") or []
        return {
            "success": True,
            "date": result.get("date"),
            "count": result.get("count", len(emails)),
            "emails": emails,
            "message": None if emails else "No emails found for today.",
        }
