"""APScheduler job definitions and lifecycle helpers."""

from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import ConfigurationError, Settings
from app.mcp.client import MCPClient
from app.services.briefing import BriefingError, send_evening_briefing

logger = logging.getLogger(__name__)


def create_scheduler(
    settings: Settings,
    mcp_client: MCPClient,
) -> AsyncIOScheduler:
    """Create and configure the application scheduler."""
    try:
        tz = ZoneInfo(settings.timezone)
    except ZoneInfoNotFoundError as exc:
        raise ConfigurationError(
            f"Invalid TIMEZONE '{settings.timezone}'. "
            "Use an IANA timezone name (e.g. Asia/Kolkata)."
        ) from exc

    hour, minute = settings.briefing_hour_minute()

    async def evening_briefing_job() -> None:
        """
        Scheduled evening briefing.

        Errors are logged and swallowed so a Gmail/Telegram failure
        does not stop future scheduled runs.
        """
        logger.info("Evening briefing job triggered")
        try:
            await send_evening_briefing(mcp_client, settings)
        except BriefingError as exc:
            logger.error("Evening briefing failed: %s", exc)
        except Exception:
            logger.exception("Evening briefing failed unexpectedly")

    scheduler = AsyncIOScheduler(timezone=tz)
    scheduler.add_job(
        evening_briefing_job,
        trigger=CronTrigger(hour=hour, minute=minute, timezone=tz),
        id="evening_briefing",
        name="Evening briefing",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    logger.info(
        "Scheduled evening briefing daily at %02d:%02d (%s)",
        hour,
        minute,
        settings.timezone,
    )
    return scheduler
