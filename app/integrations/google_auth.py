"""Google OAuth helpers and local token storage."""

from __future__ import annotations

import hashlib
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
SCOPES = [GMAIL_READONLY_SCOPE]

# Single-user local CSRF state for the in-progress OAuth flow. The PKCE
# code_verifier generated when building the authorization URL must be replayed
# at token exchange, otherwise Google rejects the code with invalid_grant.
_pending_oauth_state: str | None = None
_pending_code_verifier: str | None = None


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
    global _pending_oauth_state, _pending_code_verifier

    settings = settings or get_settings()
    flow = create_oauth_flow(settings)
    # Deliberately omit include_granted_scopes so we only ever request
    # gmail.readonly and don't drag in unrelated previously-granted scopes.
    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )
    _pending_oauth_state = state
    # authorization_url() generates the PKCE verifier as a side effect.
    _pending_code_verifier = flow.code_verifier
    logger.info("Google OAuth authorization URL generated")
    return auth_url, state


def exchange_code_for_tokens(
    authorization_response: str,
    state: str | None = None,
    settings: Settings | None = None,
) -> Credentials:
    """Exchange an OAuth callback response for credentials and persist them."""
    global _pending_oauth_state, _pending_code_verifier

    settings = settings or get_settings()
    if state is not None and _pending_oauth_state is not None and state != _pending_oauth_state:
        raise GoogleAuthError("Invalid OAuth state. Start again at /auth/google.")

    if _pending_code_verifier is None:
        raise GoogleAuthError(
            "No OAuth flow is in progress on this server process, so the PKCE "
            "verifier is unavailable. Start again at /auth/google."
        )

    try:
        flow = create_oauth_flow(settings, state=state, code_verifier=_pending_code_verifier)
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
    _pending_oauth_state = None
    _pending_code_verifier = None
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


def get_valid_credentials(settings: Settings | None = None) -> Credentials:
    """
    Return usable Google credentials, refreshing the access token when needed.

    Never logs access or refresh tokens.
    """
    settings = settings or get_settings()
    credentials = load_credentials(settings.google_token_file(), SCOPES)

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
