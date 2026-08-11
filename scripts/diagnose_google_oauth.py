"""
Diagnose Google OAuth client credentials without exposing secrets.

Sends a deliberately invalid authorization code to Google's token endpoint.
Google's error tells us whether the client_id/client_secret pair itself is valid:

    invalid_client -> the client_id/client_secret pair is wrong
    invalid_grant  -> credentials are accepted, so the problem is the code/redirect_uri
"""

from __future__ import annotations

import httpx

from app.config import get_settings

TOKEN_URI = "https://oauth2.googleapis.com/token"


def _mask(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]} (len={len(value)})"


def main() -> None:
    settings = get_settings()
    client_id = settings.google_client_id or ""
    client_secret = settings.google_client_secret or ""

    print("=== credential shape ===")
    print(f"client_id     : {_mask(client_id)}")
    print(f"  ends with .apps.googleusercontent.com: {client_id.endswith('.apps.googleusercontent.com')}")
    print(f"  has whitespace/quotes: {client_id != client_id.strip().strip(chr(34)).strip(chr(39))}")
    print(f"client_secret : {_mask(client_secret)}")
    print(f"  starts with GOCSPX-: {client_secret.startswith('GOCSPX-')}")
    print(f"  has whitespace/quotes: {client_secret != client_secret.strip().strip(chr(34)).strip(chr(39))}")
    print(f"redirect_uri  : {settings.google_redirect_uri}")

    print("\n=== live check against Google token endpoint ===")
    response = httpx.post(
        TOKEN_URI,
        data={
            "code": "deliberately-invalid-code-for-diagnostics",
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": settings.google_redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=30.0,
    )
    body = response.json()
    error = body.get("error")
    print(f"HTTP status : {response.status_code}")
    print(f"error       : {error}")
    print(f"description : {body.get('error_description')}")

    print("\n=== verdict ===")
    if error == "invalid_client":
        print("ROOT CAUSE: client_id / client_secret pair is INVALID or mismatched.")
        print("Regenerate the secret (or re-copy both) from the SAME OAuth client.")
    elif error == "invalid_grant":
        print("Credentials are VALID and accepted by Google.")
        print("So invalid_grant during real login is about the code or redirect_uri, not the client.")
    else:
        print(f"Unexpected response: {body}")


if __name__ == "__main__":
    main()
