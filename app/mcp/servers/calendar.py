"""Calendar MCP server — read-only tools backed by CalendarService."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.integrations.calendar import CalendarError, CalendarService
from app.integrations.google_auth import GoogleAuthError, GoogleAuthRequiredError
from app.mcp import MCPError, MCPServer, MCPTool

logger = logging.getLogger(__name__)


class CalendarMCPServer(MCPServer):
    """MCP tool surface for Google Calendar."""

    name = "calendar"

    def __init__(self, calendar_service: CalendarService | None = None) -> None:
        self._calendar_service = calendar_service

    def _service(self) -> CalendarService:
        return self._calendar_service or CalendarService()

    def list_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name="calendar.get_today_events",
                description="Return today's calendar events for the connected Google account (read-only).",
                handler=self.get_today_events,
                input_schema={"type": "object", "properties": {}},
            ),
            MCPTool(
                name="calendar.get_upcoming_events",
                description="Return upcoming calendar events for the next N days (default 7, max 30).",
                handler=self.get_upcoming_events,
                input_schema={
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 30,
                            "description": "Number of days ahead to include",
                        }
                    },
                },
            ),
        ]

    async def get_today_events(self) -> dict[str, Any]:
        try:
            result = await asyncio.to_thread(self._service().get_today_events)
        except GoogleAuthRequiredError as exc:
            raise MCPError(str(exc)) from exc
        except GoogleAuthError as exc:
            raise MCPError(str(exc)) from exc
        except CalendarError as exc:
            raise MCPError(str(exc)) from exc
        except Exception as exc:
            logger.error("Calendar MCP unexpected error: %s", type(exc).__name__)
            raise MCPError("calendar.get_today_events failed unexpectedly") from exc

        events = result.get("events") or []
        return {
            "success": True,
            "date": result.get("date"),
            "count": result.get("count", len(events)),
            "events": events,
            "message": None if events else "No events found for today.",
        }

    async def get_upcoming_events(self, days: int = 7) -> dict[str, Any]:
        try:
            result = await asyncio.to_thread(
                self._service().get_upcoming_events, days=days
            )
        except GoogleAuthRequiredError as exc:
            raise MCPError(str(exc)) from exc
        except GoogleAuthError as exc:
            raise MCPError(str(exc)) from exc
        except CalendarError as exc:
            raise MCPError(str(exc)) from exc
        except Exception as exc:
            logger.error("Calendar MCP unexpected error: %s", type(exc).__name__)
            raise MCPError("calendar.get_upcoming_events failed unexpectedly") from exc

        events = result.get("events") or []
        return {
            "success": True,
            "from": result.get("from"),
            "to": result.get("to"),
            "days": result.get("days", days),
            "count": result.get("count", len(events)),
            "events": events,
            "message": None if events else "No upcoming events found.",
        }
