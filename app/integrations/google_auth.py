"""Google OAuth helpers and local token storage."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

from app.config import ConfigurationError, Settings, get_settings

logger = logging.getLogger(__name__)

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
CALENDAR_READONLY_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
YOUTUBE_READONLY_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
SCOPES = [
    GMAIL_READONLY_SCOPE,
    CALENDAR_READONLY_SCOPE,
    YOUTUBE_READONLY_SCOPE,
]

# Persist PKCE verifier + state to disk so uvicorn --reload (or multi-process)
# cannot wipe in-memory globals between /auth/google and the callback.
PENDING_OAUTH_PATH = Path("tokens/google_oauth_pending.json")


def _allow_insecure_transport_for_local_dev(redirect_uri: str) -> None:
    """Allow http://localhost OAuth redirects during local development."""
    if redirect_uri.startswith("http://localhost") or redirect_uri.startswith("http://127.0.0.1"):
        os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
    # Google may return a superset of the requested scopes (e.g. previously
    # granted scopes). Relax the strict scope-equality check so this does not
    # raise instead of returning a usable gmail.readonly token.
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")


def _code_fingerprint(authorization_response: str) -> str:
    """Short non-reversible fingerprint of the auth code, for debugging reuse."""
    code = parse_qs(urlparse(authorization_response).query).get("code", [""])[0]
    if not code:
        return "none"
    return hashlib.sha256(code.encode()).hexdigest()[:8]


def _save_pending_oauth(state: str, code_verifier: str) -> None:
    PENDING_OAUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    PENDING_OAUTH_PATH.write_text(
        json.dumps({"state": state, "code_verifier": code_verifier}),
        encoding="utf-8",
    )


def _load_pending_oauth() -> tuple[str | None, str | None]:
    if not PENDING_OAUTH_PATH.exists():
        return None, None
    try:
        data = json.loads(PENDING_OAUTH_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    state = data.get("state")
    verifier = data.get("code_verifier")
    return (
        state if isinstance(state, str) else None,
        verifier if isinstance(verifier, str) else None,
    )


def _clear_pending_oauth() -> None:
    try:
        PENDING_OAUTH_PATH.unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to clear pending OAuth file")


class GoogleAuthError(Exception):
    """Raised when Google OAuth or token handling fails."""


class GoogleAuthRequiredError(GoogleAuthError):
    """Raised when the user must complete /auth/google again."""


def _client_config(settings: Settings) -> dict:
    client_id, client_secret = settings.require_google()
    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [settings.google_redirect_uri],
        }
    }


def create_oauth_flow(
    settings: Settings | None = None,
    state: str | None = None,
    code_verifier: str | None = None,
) -> Flow:
    """Create a Google OAuth Flow for the web redirect flow."""
    settings = settings or get_settings()
    _allow_insecure_transport_for_local_dev(settings.google_redirect_uri)
    return Flow.from_client_config(
        _client_config(settings),
        scopes=SCOPES,
        redirect_uri=settings.google_redirect_uri,
        state=state,
        code_verifier=code_verifier,
    )


def build_authorization_url(settings: Settings | None = None) -> tuple[str, str]:
    """
    Start OAuth and return (authorization_url, state).

    Requests offline access so a refresh token is issued.
    """
    settings = settings or get_settings()
    flow = create_oauth_flow(settings)
    # Deliberately omit include_granted_scopes so we only request the
    # explicitly configured read-only Gmail/Calendar/YouTube scopes.
    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )
    if not flow.code_verifier:
        raise GoogleAuthError("Failed to generate PKCE verifier for Google OAuth.")
    _save_pending_oauth(state=state, code_verifier=flow.code_verifier)
    logger.info("Google OAuth authorization URL generated")
    return auth_url, state


def exchange_code_for_tokens(
    authorization_response: str,
    state: str | None = None,
    settings: Settings | None = None,
) -> Credentials:
    """Exchange an OAuth callback response for credentials and persist them."""
    settings = settings or get_settings()
    pending_state, pending_verifier = _load_pending_oauth()

    if state is not None and pending_state is not None and state != pending_state:
        raise GoogleAuthError("Invalid OAuth state. Start again at /auth/google.")

    if not pending_verifier:
        raise GoogleAuthError(
            "No OAuth flow is in progress (PKCE verifier missing). "
            "Start again at /auth/google."
        )

    try:
        flow = create_oauth_flow(settings, state=state, code_verifier=pending_verifier)
        logger.info(
            "Exchanging Google authorization code (code_fp=%s, pkce=yes)",
            _code_fingerprint(authorization_response),
        )
        flow.fetch_token(authorization_response=authorization_response)
    except ConfigurationError:
        raise
    except Exception as exc:
        # Surface Google's real reason (e.g. invalid_grant) to logs to aid
        # debugging, without ever logging tokens.
        reason = getattr(exc, "error", None) or getattr(exc, "description", None)
        logger.error(
            "Google OAuth token exchange failed: %s (%s)",
            type(exc).__name__,
            reason or "no detail",
        )
        hint = (
            "Google rejected the authorization code (invalid_grant). "
            "This usually means the code was already used/expired or your "
            "system clock is out of sync. Start fresh at /auth/google."
            if reason == "invalid_grant"
            else "Google OAuth failed while exchanging the authorization code. "
            "Try again at /auth/google."
        )
        raise GoogleAuthError(hint) from exc

    credentials = flow.credentials
    save_credentials(credentials, settings.google_token_file())
    _clear_pending_oauth()
    logger.info("Google OAuth credentials saved locally")
    return credentials


def save_credentials(credentials: Credentials, token_path: Path | str) -> None:
    """Persist OAuth credentials to a local JSON file (never log secrets)."""
    path = Path(token_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(credentials.to_json(), encoding="utf-8")
    logger.info("Wrote Google credentials to %s", path)


def load_credentials(token_path: Path | str, scopes: list[str] | None = None) -> Credentials:
    """Load credentials from disk. Does not refresh."""
    path = Path(token_path)
    if not path.exists():
        raise GoogleAuthRequiredError(
            "Google account is not connected. Open /auth/google to authenticate."
        )
    try:
        return Credentials.from_authorized_user_file(str(path), scopes or SCOPES)
    except Exception as exc:
        logger.error("Failed to load Google credentials file: %s", type(exc).__name__)
        raise GoogleAuthError(
            "Stored Google credentials are invalid. Open /auth/google to authenticate again."
        ) from exc


def missing_scopes(credentials: Credentials, required: list[str] | None = None) -> list[str]:
    """Return required scopes that are not present on the credential."""
    needed = required or SCOPES
    raw = credentials.scopes or []
    if not isinstance(raw, (list, tuple, set)):
        return list(needed)
    granted = {s for s in raw if isinstance(s, str) and s}
    return [scope for scope in needed if scope not in granted]


def get_valid_credentials(settings: Settings | None = None) -> Credentials:
    """
    Return usable Google credentials, refreshing the access token when needed.

    Never logs access or refresh tokens.
    """
    settings = settings or get_settings()
    credentials = load_credentials(settings.google_token_file(), SCOPES)

    missing = missing_scopes(credentials, SCOPES)
    if missing:
        logger.warning(
            "Stored Google credentials are missing %s required scope(s); re-auth required",
            len(missing),
        )
        raise GoogleAuthRequiredError(
            "Google credentials are missing Calendar/YouTube (or Gmail) scopes. "
            "Open /auth/google to re-authenticate and grant the new permissions."
        )

    if credentials.valid:
        return credentials

    if credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
        except RefreshError as exc:
            logger.error("Google refresh token rejected; re-authentication required")
            raise GoogleAuthRequiredError(
                "Google refresh token is invalid or revoked. "
                "Open /auth/google to authenticate again."
            ) from exc
        except Exception as exc:
            logger.error("Google token refresh failed: %s", type(exc).__name__)
            raise GoogleAuthError(
                "Failed to refresh Google access token. Try /auth/google again."
            ) from exc

        save_credentials(credentials, settings.google_token_file())
        logger.info("Google access token refreshed")
        return credentials

    raise GoogleAuthRequiredError(
        "Google credentials are missing a refresh token or are no longer valid. "
        "Open /auth/google to authenticate again."
    )


def has_stored_credentials(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return settings.google_token_file().exists()
