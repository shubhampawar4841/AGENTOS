"""YouTube MCP server — read-only tools backed by YouTubeService."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.integrations.google_auth import GoogleAuthError, GoogleAuthRequiredError
from app.integrations.youtube import YouTubeError, YouTubeService
from app.mcp import MCPError, MCPServer, MCPTool

logger = logging.getLogger(__name__)


class YouTubeMCPServer(MCPServer):
    """MCP tool surface for YouTube (configured favorite channels)."""

    name = "youtube"

    def __init__(self, youtube_service: YouTubeService | None = None) -> None:
        self._youtube_service = youtube_service

    def _service(self) -> YouTubeService:
        return self._youtube_service or YouTubeService()

    def list_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name="youtube.get_recent_videos",
                description=(
                    "Return recent uploads from configured favorite YouTube channels "
                    "(YOUTUBE_CHANNEL_IDS). Read-only."
                ),
                handler=self.get_recent_videos,
                input_schema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 25,
                            "description": "Max videos to return",
                        }
                    },
                },
            )
        ]

    async def get_recent_videos(self, limit: int = 10) -> dict[str, Any]:
        try:
            result = await asyncio.to_thread(
                self._service().get_recent_videos, limit=limit
            )
        except GoogleAuthRequiredError as exc:
            raise MCPError(str(exc)) from exc
        except GoogleAuthError as exc:
            raise MCPError(str(exc)) from exc
        except YouTubeError as exc:
            raise MCPError(str(exc)) from exc
        except Exception as exc:
            logger.error("YouTube MCP unexpected error: %s", type(exc).__name__)
            raise MCPError("youtube.get_recent_videos failed unexpectedly") from exc

        videos = result.get("videos") or []
        payload = {
            "success": True,
            "count": result.get("count", len(videos)),
            "videos": videos,
        }
        if result.get("message"):
            payload["message"] = result["message"]
        elif not videos:
            payload["message"] = "No recent videos found."
        return payload
