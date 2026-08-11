"""Milestone 2 tests: Google OAuth tokens, Gmail normalization, MCP, /test/gmail."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import ConfigurationError, get_settings
from app.integrations.gmail import GmailService, normalize_gmail_message
from app.integrations.google_auth import (
    GoogleAuthRequiredError,
    get_valid_credentials,
    load_credentials,
    save_credentials,
)
from app.mcp import MCPError
from app.mcp.client import MCPClient
from app.mcp.servers.gmail import GmailMCPServer


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings_env(monkeypatch, tmp_path):
    token_path = tmp_path / "google_token.json"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("TIMEZONE", "Asia/Kolkata")
    monkeypatch.setenv("BRIEFING_TIME", "19:00")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv(
        "GOOGLE_REDIRECT_URI", "http://localhost:3000/auth/google/callback"
    )
    monkeypatch.setenv("GOOGLE_TOKEN_PATH", str(token_path))
    get_settings.cache_clear()
    return token_path


def test_require_google_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    monkeypatch.setenv("TIMEZONE", "Asia/Kolkata")
    monkeypatch.setenv("BRIEFING_TIME", "19:00")
    get_settings.cache_clear()

    with pytest.raises(ConfigurationError, match="GOOGLE_CLIENT_ID"):
        get_settings().require_google()


def test_pkce_verifier_survives_into_token_exchange(settings_env, tmp_path, monkeypatch):
    """
    Google enables PKCE by default in google-auth-oauthlib. The verifier created
    while building the authorization URL must be replayed at token exchange,
    otherwise Google rejects the code with invalid_grant.
    """
    import hashlib
    from base64 import urlsafe_b64encode
    from urllib.parse import parse_qs, urlparse

    import app.integrations.google_auth as ga

    monkeypatch.setattr(ga, "PENDING_OAUTH_PATH", tmp_path / "pending.json")

    url, state = ga.build_authorization_url()
    query = parse_qs(urlparse(url).query)

    assert query["code_challenge_method"][0] == "S256"
    sent_challenge = query["code_challenge"][0]

    pending_state, verifier = ga._load_pending_oauth()
    assert pending_state == state
    assert verifier is not None

    digest = hashlib.sha256(verifier.encode()).digest()
    recomputed = urlsafe_b64encode(digest).decode().split("=")[0]
    assert recomputed == sent_challenge

    flow = ga.create_oauth_flow(state=state, code_verifier=verifier)
    assert flow.code_verifier == verifier


def test_exchange_without_pending_flow_is_rejected(settings_env, tmp_path, monkeypatch):
    import app.integrations.google_auth as ga

    monkeypatch.setattr(ga, "PENDING_OAUTH_PATH", tmp_path / "pending.json")
    ga._clear_pending_oauth()
    with pytest.raises(Exception, match="/auth/google"):
        ga.exchange_code_for_tokens(
            authorization_response="http://localhost:3000/auth/google/callback?code=x&state=y"
        )


def test_load_credentials_missing(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(GoogleAuthRequiredError, match="/auth/google"):
        load_credentials(missing)


def test_save_and_load_credentials(tmp_path):
    path = tmp_path / "google_token.json"
    creds = MagicMock()
    creds.to_json.return_value = json.dumps(
        {
            "token": "access-token",
            "refresh_token": "refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
        }
    )

    save_credentials(creds, path)
    assert path.exists()
    loaded = load_credentials(path)
    assert loaded.token == "access-token"
    assert loaded.refresh_token == "refresh-token"


def test_get_valid_credentials_missing_file(settings_env):
    with pytest.raises(GoogleAuthRequiredError):
        get_valid_credentials(get_settings())


def test_get_valid_credentials_refreshes_expired(settings_env, tmp_path):
    from app.integrations.google_auth import SCOPES

    path = settings_env
    path.write_text(
        json.dumps(
            {
                "token": "old-access",
                "refresh_token": "refresh-token",
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": "client-id",
                "client_secret": "client-secret",
                "scopes": SCOPES,
                "expiry": "2000-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    refreshed = MagicMock()
    refreshed.valid = True
    refreshed.expired = False
    refreshed.refresh_token = "refresh-token"
    refreshed.to_json.return_value = json.dumps(
        {
            "token": "new-access",
            "refresh_token": "refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "scopes": SCOPES,
        }
    )

    with patch(
        "app.integrations.google_auth.load_credentials",
        return_value=MagicMock(
            valid=False,
            expired=True,
            refresh_token="refresh-token",
            scopes=list(SCOPES),
            refresh=MagicMock(side_effect=lambda _req: setattr(refreshed, "valid", True)),
        ),
    ) as load_mock:
        # Make refresh mutate the same object into a valid credential we can save.
        cred = load_mock.return_value

        def _refresh(_request):
            cred.valid = True
            cred.expired = False
            cred.to_json = refreshed.to_json

        cred.refresh.side_effect = _refresh
        result = get_valid_credentials(get_settings())

    assert result.valid is True
    assert path.exists()


def test_normalize_gmail_message():
    raw = {
        "id": "abc",
        "threadId": "thread-1",
        "snippet": "Hello world",
        "internalDate": "1691731200000",
        "labelIds": ["INBOX", "UNREAD"],
        "payload": {
            "headers": [
                {"name": "From", "value": "Alice <alice@example.com>"},
                {"name": "Subject", "value": "Weekly sync"},
            ]
        },
    }
    normalized = normalize_gmail_message(raw)
    assert normalized == {
        "id": "abc",
        "thread_id": "thread-1",
        "sender": "Alice <alice@example.com>",
        "subject": "Weekly sync",
        "snippet": "Hello world",
        "timestamp": normalized["timestamp"],
        "labels": ["INBOX", "UNREAD"],
    }
    assert normalized["timestamp"]


def test_normalize_gmail_message_missing_headers():
    normalized = normalize_gmail_message({"id": "x", "payload": {}})
    assert normalized["id"] == "x"
    assert normalized["sender"] == ""
    assert normalized["subject"] == ""
    assert normalized["labels"] == []


@pytest.mark.asyncio
async def test_mcp_gmail_tool_schema(settings_env):
    mock_service = MagicMock()
    mock_service.get_today_emails.return_value = {
        "date": "2026-08-11",
        "count": 1,
        "emails": [
            {
                "id": "1",
                "thread_id": "t1",
                "sender": "bob@example.com",
                "subject": "Invoice",
                "snippet": "Please find...",
                "timestamp": "2026-08-11T12:00:00",
                "labels": ["INBOX"],
            }
        ],
    }
    server = GmailMCPServer(gmail_service=mock_service)
    result = await server.get_today_emails()

    assert result["success"] is True
    assert result["count"] == 1
    email = result["emails"][0]
    assert set(email.keys()) >= {
        "id",
        "thread_id",
        "sender",
        "subject",
        "snippet",
        "timestamp",
        "labels",
    }


@pytest.mark.asyncio
async def test_mcp_gmail_tool_auth_error():
    mock_service = MagicMock()
    mock_service.get_today_emails.side_effect = GoogleAuthRequiredError(
        "Open /auth/google to authenticate."
    )
    server = GmailMCPServer(gmail_service=mock_service)
    with pytest.raises(MCPError, match="/auth/google"):
        await server.get_today_emails()


def test_test_gmail_endpoint_success(settings_env):
    from app.main import app

    fake_result = {
        "success": True,
        "date": "2026-08-11",
        "count": 1,
        "emails": [
            {
                "id": "1",
                "thread_id": "t1",
                "sender": "a@example.com",
                "subject": "Hi",
                "snippet": "Hello",
                "timestamp": "2026-08-11T10:00:00",
                "labels": ["INBOX"],
            }
        ],
    }

    with patch.object(MCPClient, "call_tool", return_value=fake_result) as mocked:
        # call_tool is async
        async def _call(*_args, **_kwargs):
            return fake_result

        mocked.side_effect = _call
        with TestClient(app) as client:
            response = client.get("/test/gmail")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["emails"][0]["subject"] == "Hi"
    assert "refresh_token" not in response.text
    assert "client-secret" not in response.text
    assert "access-token" not in response.text


def test_test_gmail_endpoint_auth_required(settings_env):
    from app.main import app

    async def _call(*_args, **_kwargs):
        raise MCPError("Google account is not connected. Open /auth/google to authenticate.")

    with patch.object(MCPClient, "call_tool", side_effect=_call):
        with TestClient(app) as client:
            response = client.get("/test/gmail")

    assert response.status_code == 401
    assert "refresh_token" not in response.text


def test_gmail_service_uses_normalized_data(settings_env):
    settings = get_settings()
    service = GmailService(settings=settings)

    mock_creds = MagicMock()
    mock_api = MagicMock()
    mock_api.users.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": [{"id": "m1"}]
    }
    mock_api.users.return_value.messages.return_value.get.return_value.execute.return_value = {
        "id": "m1",
        "threadId": "t1",
        "snippet": "Snippet",
        "internalDate": "1691731200000",
        "labelIds": ["INBOX"],
        "payload": {
            "headers": [
                {"name": "From", "value": "Ada <ada@example.com>"},
                {"name": "Subject", "value": "Hello"},
            ]
        },
    }

    with (
        patch("app.integrations.gmail.get_valid_credentials", return_value=mock_creds),
        patch("app.integrations.gmail.build", return_value=mock_api),
    ):
        result = service.get_today_emails(max_results=5)

    assert result["count"] == 1
    assert result["emails"][0]["sender"] == "Ada <ada@example.com>"
    assert result["emails"][0]["subject"] == "Hello"
    assert "payload" not in result["emails"][0]
