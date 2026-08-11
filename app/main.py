"""FastAPI application entrypoint for Personal Agentic OS."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.agent import PersonalAgent, process_message
from app.config import ConfigurationError, get_settings
from app.integrations.google_auth import (
    GoogleAuthError,
    build_authorization_url,
    exchange_code_for_tokens,
)
from app.integrations.telegram import TelegramError, TelegramPoller, TelegramService
from app.mcp import MCPError
from app.mcp.client import MCPClient, create_default_mcp_client
from app.scheduler.jobs import create_scheduler
from app.services.briefing import BriefingError, send_evening_briefing
from app.services.llm import LLMService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    scheduler: AsyncIOScheduler | None = None
    poller: TelegramPoller | None = None

    app.state.mcp_client = create_default_mcp_client()
    app.state.llm = LLMService(settings)
    app.state.agent = PersonalAgent(app.state.mcp_client, app.state.llm)
    logger.info(
        "MCP client ready with tools: %s",
        [t.name for t in app.state.mcp_client.list_tools()],
    )

    try:
        scheduler = create_scheduler(settings, app.state.mcp_client)
        scheduler.start()
        app.state.scheduler = scheduler
        logger.info("Scheduler started")
    except ConfigurationError as exc:
        logger.error("Scheduler configuration error: %s", exc)
        raise

    if settings.telegram_configured:
        telegram = TelegramService.from_settings(settings)

        async def _handle(text: str) -> str:
            return await process_message(text, app.state.agent)

        poller = TelegramPoller(telegram, _handle)
        app.state.telegram_poller = poller
        await poller.start()
    else:
        app.state.telegram_poller = None
        logger.warning(
            "Telegram polling not started "
            "(set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)"
        )

    try:
        yield
    finally:
        if poller is not None:
            await poller.stop()
        if scheduler is not None and scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("Scheduler shut down")


app = FastAPI(
    title="Personal Agentic OS",
    description="Single-user personal AI assistant foundation",
    version="0.4.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/google")
async def auth_google_start() -> RedirectResponse:
    """Begin Google OAuth (Gmail readonly)."""
    try:
        auth_url, _state = build_authorization_url(get_settings())
    except ConfigurationError as exc:
        logger.error("Google OAuth configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GoogleAuthError as exc:
        logger.error("Google OAuth start failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return RedirectResponse(url=auth_url, status_code=302)


@app.get("/auth/google/callback")
async def auth_google_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> JSONResponse:
    """Handle Google OAuth callback and store tokens locally."""
    if error:
        logger.error("Google OAuth provider returned an error")
        raise HTTPException(
            status_code=400,
            detail=f"Google OAuth was denied or failed ({error}). Try /auth/google again.",
        )
    if not code:
        raise HTTPException(
            status_code=400,
            detail="Missing authorization code. Start again at /auth/google.",
        )

    try:
        exchange_code_for_tokens(
            authorization_response=str(request.url),
            state=state,
            settings=get_settings(),
        )
    except ConfigurationError as exc:
        logger.error("Google OAuth configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GoogleAuthError as exc:
        logger.error("Google OAuth callback failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(
        {
            "success": True,
            "message": "Google account connected. You can now call GET /test/gmail.",
        }
    )


@app.post("/test/telegram")
async def test_telegram() -> JSONResponse:
    """Send a test message to the configured Telegram chat."""
    settings = get_settings()
    try:
        telegram = TelegramService.from_settings(settings)
        await telegram.send_message("🤖 Personal Agent is working!")
    except ConfigurationError as exc:
        logger.error("Telegram configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TelegramError as exc:
        # Message must never include the bot token.
        logger.error("Telegram send failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return JSONResponse({"status": "ok", "message": "Test Telegram message sent"})


@app.get("/test/gmail")
async def test_gmail() -> JSONResponse:
    """Fetch today's emails through the MCP Gmail tool."""
    client: MCPClient = app.state.mcp_client
    try:
        result = await client.call_tool("gmail.get_today_emails")
    except MCPError as exc:
        message = str(exc)
        logger.error("MCP Gmail tool failed: %s", message)
        lower = message.lower()
        if "authenticate" in lower or "not connected" in lower or "revoked" in lower:
            raise HTTPException(status_code=401, detail=message) from exc
        raise HTTPException(status_code=502, detail=message) from exc

    emails = result.get("emails") if isinstance(result, dict) else []
    payload = {
        "success": True,
        "date": result.get("date") if isinstance(result, dict) else None,
        "count": result.get("count", len(emails or [])) if isinstance(result, dict) else 0,
        "emails": emails or [],
    }
    if isinstance(result, dict) and result.get("message"):
        payload["message"] = result["message"]
    return JSONResponse(payload)


@app.post("/test/briefing")
async def test_briefing() -> JSONResponse:
    """Run the evening briefing workflow immediately (same path as the scheduler)."""
    client: MCPClient = app.state.mcp_client
    try:
        await send_evening_briefing(client, get_settings())
    except BriefingError as exc:
        message = str(exc)
        logger.error("Manual briefing failed: %s", message)
        lower = message.lower()
        if "authenticate" in lower or "not connected" in lower or "revoked" in lower:
            raise HTTPException(status_code=401, detail=message) from exc
        if "telegram is not configured" in lower:
            raise HTTPException(status_code=503, detail=message) from exc
        raise HTTPException(status_code=502, detail=message) from exc

    return JSONResponse(
        {
            "status": "ok",
            "message": "Evening briefing sent to Telegram",
        }
    )


@app.post("/test/agent")
async def test_agent(payload: dict[str, Any] | None = None) -> JSONResponse:
    """Process a message through the agent without Telegram polling."""
    message = ""
    if isinstance(payload, dict):
        message = str(payload.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Provide JSON {\"message\": \"...\"}")

    agent: PersonalAgent = app.state.agent
    reply = await process_message(message, agent)
    return JSONResponse({"status": "ok", "reply": reply})


@app.get("/mcp/tools")
async def list_mcp_tools() -> dict[str, Any]:
    """Discover registered MCP tools."""
    client: MCPClient = app.state.mcp_client
    tools = [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in client.list_tools()
    ]
    return {"tools": tools}


@app.post("/mcp/tools/{tool_name:path}/call")
async def call_mcp_tool(tool_name: str) -> dict[str, Any]:
    """Invoke an MCP tool by name."""
    client: MCPClient = app.state.mcp_client
    try:
        result = await client.call_tool(tool_name)
    except MCPError as exc:
        logger.error("MCP tool error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"tool": tool_name, "result": result}
