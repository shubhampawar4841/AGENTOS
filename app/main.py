"""FastAPI application entrypoint for Personal Agentic OS."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Header, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.agent import PersonalAgent, process_message
from app.agent.agent import run_agent_turn
from app.agent.state import AgentRunResult, ConversationStore
from app.config import ConfigurationError, Settings, get_settings
from app.integrations.google_auth import (
    GoogleAuthError,
    build_authorization_url,
    exchange_code_for_tokens,
    export_credentials_json,
)
from app.integrations.telegram import (
    TELEGRAM_SECRET_HEADER,
    TelegramError,
    TelegramPoller,
    TelegramService,
    handle_update,
    validate_webhook_secret,
)
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


def _build_telegram_message_handler(app: FastAPI):
    async def _handle(text: str, chat_id: str) -> str:
        async def _run(
            current_message: str,
            history: list[dict[str, str]],
            verified: dict[str, Any],
        ) -> AgentRunResult:
            return await run_agent_turn(
                current_message,
                app.state.agent,
                conversation_history=history,
                verified_context=verified,
            )

        return await app.state.conversation_store.process_turn(
            chat_id,
            text,
            _run,
        )

    return _handle


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    scheduler: AsyncIOScheduler | None = None
    poller: TelegramPoller | None = None

    app.state.mcp_client = create_default_mcp_client()
    app.state.llm = LLMService(settings)
    app.state.agent = PersonalAgent(app.state.mcp_client, app.state.llm)
    app.state.conversation_store = ConversationStore(max_messages=20)
    app.state.telegram = None
    app.state.telegram_message_handler = None
    app.state.telegram_poller = None
    logger.info(
        "MCP client ready with tools: %s",
        [t.name for t in app.state.mcp_client.list_tools()],
    )

    # Long-running schedulers are unreliable on Vercel serverless; keep local only.
    if not settings.is_serverless:
        try:
            scheduler = create_scheduler(settings, app.state.mcp_client)
            scheduler.start()
            app.state.scheduler = scheduler
            logger.info("Scheduler started")
        except ConfigurationError as exc:
            logger.error("Scheduler configuration error: %s", exc)
            raise
    else:
        app.state.scheduler = None
        logger.info("Scheduler skipped (serverless/production webhook mode)")

    transport = settings.telegram_transport
    if settings.telegram_configured:
        telegram = TelegramService.from_settings(settings)
        handler = _build_telegram_message_handler(app)
        app.state.telegram = telegram
        app.state.telegram_message_handler = handler

        # Polling is a long-running task and must never start on serverless.
        if settings.polling_enabled:
            poller = TelegramPoller(telegram, handler)
            app.state.telegram_poller = poller
            await poller.start()
            logger.info("Telegram transport=polling (local development)")
        else:
            logger.info("Telegram transport=webhook (polling disabled)")
            if not settings.telegram_webhook_configured:
                logger.warning(
                    "TELEGRAM_WEBHOOK_SECRET is not set; "
                    "webhook requests will be rejected until it is configured"
                )
    else:
        logger.warning(
            "Telegram not configured "
            "(set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID); transport=%s",
            transport,
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
    version="0.5.0",
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "SYNCOS"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/telegram/webhook/status")
async def telegram_webhook_status() -> JSONResponse:
    """Safe Telegram transport status (never returns secrets)."""
    settings = get_settings()
    return JSONResponse(
        {
            "configured": settings.telegram_configured,
            "webhook_secret_configured": bool(settings.telegram_webhook_secret),
            "mode": settings.telegram_transport,
            "mode_explicit": settings.telegram_mode is not None,
            "serverless": settings.is_serverless,
            "polling_enabled": settings.polling_enabled,
            # Note: conversation history is in-memory and ephemeral on Vercel.
            "conversation_memory": "in_memory_ephemeral",
        }
    )


async def _process_telegram_webhook(
    request: Request,
    *,
    path_secret: str | None,
    header_secret: str | None,
) -> JSONResponse:
    settings: Settings = get_settings()
    if not settings.telegram_configured:
        raise HTTPException(status_code=503, detail="Telegram is not configured")
    if not settings.telegram_webhook_secret:
        raise HTTPException(
            status_code=503,
            detail="Webhook secret is not configured. Set TELEGRAM_WEBHOOK_SECRET.",
        )
    if not validate_webhook_secret(
        settings.telegram_webhook_secret,
        path_secret=path_secret,
        header_secret=header_secret,
    ):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    try:
        update = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if not isinstance(update, dict):
        raise HTTPException(status_code=400, detail="Telegram update must be an object")

    telegram: TelegramService | None = getattr(app.state, "telegram", None)
    handler = getattr(app.state, "telegram_message_handler", None)
    if telegram is None or handler is None:
        # Cold start / incomplete lifespan — build ephemeral handlers.
        telegram = TelegramService.from_settings(settings)
        handler = _build_telegram_message_handler(app)
        app.state.telegram = telegram
        app.state.telegram_message_handler = handler

    result = await handle_update(
        update,
        telegram=telegram,
        message_handler=handler,
    )
    # Telegram retries on non-2xx; always acknowledge after processing attempt.
    return JSONResponse({"ok": True, **result})


@app.post("/api/telegram/webhook")
async def telegram_webhook_header(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(
        default=None,
        alias=TELEGRAM_SECRET_HEADER,
    ),
) -> JSONResponse:
    """Webhook endpoint authenticated via Telegram secret-token header."""
    return await _process_telegram_webhook(
        request,
        path_secret=None,
        header_secret=x_telegram_bot_api_secret_token,
    )


@app.post("/api/telegram/webhook/{webhook_secret}")
async def telegram_webhook_path(
    request: Request,
    webhook_secret: str = Path(...),
    x_telegram_bot_api_secret_token: str | None = Header(
        default=None,
        alias=TELEGRAM_SECRET_HEADER,
    ),
) -> JSONResponse:
    """Webhook endpoint authenticated via secret path segment and/or header."""
    return await _process_telegram_webhook(
        request,
        path_secret=webhook_secret,
        header_secret=x_telegram_bot_api_secret_token,
    )


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

    settings = get_settings()
    try:
        credentials = exchange_code_for_tokens(
            authorization_response=str(request.url),
            state=state,
            settings=settings,
        )
    except ConfigurationError as exc:
        logger.error("Google OAuth configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GoogleAuthError as exc:
        logger.error("Google OAuth callback failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if settings.google_token_file().exists():
        return JSONResponse(
            {
                "success": True,
                "message": "Google account connected. You can now call GET /test/gmail.",
            }
        )

    # Ephemeral filesystem (e.g. Vercel): the token cannot outlive this request,
    # so hand it to the user who just authenticated to store as GOOGLE_TOKEN_JSON.
    logger.warning(
        "Google token could not be persisted; returning it for GOOGLE_TOKEN_JSON"
    )
    return JSONResponse(
        {
            "success": True,
            "action_required": (
                "This deployment has no writable storage, so the token cannot be "
                "saved here. Copy the 'google_token_json' value below into an "
                "environment variable named GOOGLE_TOKEN_JSON, then redeploy. "
                "You only need to do this once."
            ),
            "google_token_json": credentials.to_json(),
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/auth/google/token")
async def auth_google_token() -> JSONResponse:
    """
    Return the current Google token JSON for GOOGLE_TOKEN_JSON.

    Local-only helper: it exposes a refresh token, so it is refused on
    serverless/production where the response could travel over the network.
    """
    settings = get_settings()
    if settings.is_serverless:
        raise HTTPException(
            status_code=403,
            detail="Token export is disabled in production for safety. "
            "Run this locally after /auth/google.",
        )
    try:
        token_json = export_credentials_json(settings)
    except GoogleAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(
        {
            "instructions": (
                "Copy the value of 'google_token_json' into a Vercel environment "
                "variable named GOOGLE_TOKEN_JSON, then redeploy."
            ),
            "google_token_json": token_json,
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
    conversation_id = "test-agent"
    if isinstance(payload, dict) and payload.get("conversation_id") is not None:
        conversation_id = str(payload["conversation_id"])

    async def _run(
        current_message: str,
        history: list[dict[str, str]],
        verified: dict[str, Any],
    ) -> str:
        return await process_message(
            current_message,
            agent,
            conversation_history=history,
            verified_context=verified,
        )

    store: ConversationStore = app.state.conversation_store
    reply = await store.process_turn(conversation_id, message, _run)
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
