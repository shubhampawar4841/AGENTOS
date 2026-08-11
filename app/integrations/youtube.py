"""Read-only YouTube Data API integration."""

from __future__ import annotations

import logging
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.config import Settings, get_settings
from app.integrations.google_auth import (
    GoogleAuthError,
    GoogleAuthRequiredError,
    get_valid_credentials,
)

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 10
MAX_LIMIT = 25


class YouTubeError(Exception):
    """Raised when the YouTube API fails or returns unexpected data."""


def _truncate(text: str, limit: int = 180) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def normalize_youtube_video(
    *,
    video_id: str,
    title: str,
    channel: str,
    published_at: str,
    description: str,
) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "title": title or "(no title)",
        "channel": channel or "",
        "published_at": published_at or "",
        "description": _truncate(description),
        "url": f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
    }


class YouTubeService:
    """Thin read-only wrapper around the YouTube Data API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _build(self):
        credentials = get_valid_credentials(self.settings)
        return build("youtube", "v3", credentials=credentials, cache_discovery=False)

    def get_my_channel(self) -> dict[str, Any]:
        """Return the authenticated user's channel metadata (read-only)."""
        try:
            service = self._build()
            result = (
                service.channels()
                .list(part="snippet,contentDetails", mine=True, maxResults=1)
                .execute()
            )
        except HttpError as exc:
            status = getattr(exc.resp, "status", None)
            logger.error("YouTube channels.list(mine) failed (status=%s)", status)
            if status in {401, 403}:
                raise GoogleAuthRequiredError(
                    "YouTube access was denied. Open /auth/google to authenticate again."
                ) from exc
            raise YouTubeError("YouTube API failed while fetching your channel.") from exc
        except (GoogleAuthRequiredError, GoogleAuthError):
            raise
        except Exception as exc:
            logger.error("Unexpected YouTube channel failure: %s", type(exc).__name__)
            raise YouTubeError("Unexpected error while contacting YouTube.") from exc

        items = (result or {}).get("items") or []
        if not items:
            return {"channel_id": "", "title": "", "description": ""}
        item = items[0]
        snippet = item.get("snippet") or {}
        return {
            "channel_id": str(item.get("id") or ""),
            "title": str(snippet.get("title") or ""),
            "description": _truncate(str(snippet.get("description") or "")),
        }

    def _uploads_playlist_id(self, service, channel_id: str) -> str | None:
        result = (
            service.channels()
            .list(part="contentDetails,snippet", id=channel_id, maxResults=1)
            .execute()
        )
        items = (result or {}).get("items") or []
        if not items:
            return None
        related = ((items[0].get("contentDetails") or {}).get("relatedPlaylists") or {})
        return related.get("uploads")

    def _recent_from_playlist(
        self,
        service,
        playlist_id: str,
        *,
        channel_title: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        result = (
            service.playlistItems()
            .list(
                part="snippet,contentDetails",
                playlistId=playlist_id,
                maxResults=limit,
            )
            .execute()
        )
        videos: list[dict[str, Any]] = []
        for item in result.get("items") or []:
            if not isinstance(item, dict):
                continue
            snippet = item.get("snippet") or {}
            content = item.get("contentDetails") or {}
            video_id = str(
                content.get("videoId")
                or (snippet.get("resourceId") or {}).get("videoId")
                or ""
            )
            if not video_id:
                continue
            videos.append(
                normalize_youtube_video(
                    video_id=video_id,
                    title=str(snippet.get("title") or ""),
                    channel=str(snippet.get("channelTitle") or channel_title or ""),
                    published_at=str(
                        content.get("videoPublishedAt") or snippet.get("publishedAt") or ""
                    ),
                    description=str(snippet.get("description") or ""),
                )
            )
        return videos

    def get_recent_videos(self, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
        """
        Fetch recent uploads from configured favorite channels.

        Requires YOUTUBE_CHANNEL_IDS. Does not invent videos.
        """
        limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
        channel_ids = self.settings.youtube_channel_ids
        if not channel_ids:
            return {
                "count": 0,
                "videos": [],
                "message": (
                    "No YouTube channels configured. "
                    "Set YOUTUBE_CHANNEL_IDS in your .env (comma-separated channel IDs)."
                ),
            }

        try:
            service = self._build()
        except (GoogleAuthRequiredError, GoogleAuthError):
            raise
        except Exception as exc:
            logger.error("Unexpected YouTube client failure: %s", type(exc).__name__)
            raise YouTubeError("Unexpected error while contacting YouTube.") from exc

        per_channel = max(1, limit // max(1, len(channel_ids)))
        collected: list[dict[str, Any]] = []

        try:
            for channel_id in channel_ids:
                playlist_id = self._uploads_playlist_id(service, channel_id)
                if not playlist_id:
                    logger.warning("No uploads playlist for channel_id=%s", channel_id)
                    continue
                # Resolve a friendly channel title from the playlist items themselves.
                videos = self._recent_from_playlist(
                    service,
                    playlist_id,
                    channel_title="",
                    limit=per_channel,
                )
                collected.extend(videos)
        except HttpError as exc:
            status = getattr(exc.resp, "status", None)
            logger.error("YouTube recent videos failed (status=%s)", status)
            if status in {401, 403}:
                raise GoogleAuthRequiredError(
                    "YouTube access was denied. Open /auth/google to authenticate again."
                ) from exc
            raise YouTubeError("YouTube API failed while fetching recent videos.") from exc
        except (GoogleAuthRequiredError, GoogleAuthError, YouTubeError):
            raise
        except Exception as exc:
            logger.error("Unexpected YouTube recent failure: %s", type(exc).__name__)
            raise YouTubeError("Unexpected error while fetching YouTube videos.") from exc

        collected.sort(key=lambda v: v.get("published_at") or "", reverse=True)
        videos = collected[:limit]
        return {"count": len(videos), "videos": videos, "channels": channel_ids}
