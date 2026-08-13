# AGENTOS / SYNCOS

A single-user personal AI assistant. You chat with it on Telegram; a Groq LLM
decides which read-only tools to call; those tools run through an in-process MCP
layer that talks to Gmail, Google Calendar, and YouTube.

```
Telegram → Agent → Groq (native tool calling) → MCP client → Google APIs
                     ↑                              │
                     └────── verified tool results ─┘
```

All integrations are **read-only**. The agent can read and analyze your data, but
it cannot send, delete, or modify anything.

---

## 1. Requirements

- Python 3.11+ (developed on 3.13)
- A Telegram bot token and your chat ID
- A Groq API key
- Google OAuth client credentials (for Gmail / Calendar / YouTube)

## 2. Setup

```bash
git clone <your-repo-url>
cd Personal-agent

python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
# .venv\Scripts\Activate.ps1       # Windows PowerShell
# source .venv/bin/activate        # macOS / Linux

pip install -r requirements.txt
```

Create your env file:

```bash
cp .env.example .env
```

Then fill in `.env` (see the reference in section 7).

## 3. Connect your Google account

For local OAuth, `GOOGLE_REDIRECT_URI` must point at your local server:

```env
GOOGLE_REDIRECT_URI=http://localhost:3000/auth/google/callback
```

Add that exact URI to your Google Cloud OAuth client's authorized redirect URIs,
start the server (section 4), then open:

```
http://localhost:3000/auth/google
```

Grant the read-only Gmail + Calendar + YouTube scopes. Tokens are stored in
`tokens/` (gitignored). Re-run this whenever scopes change or the token is revoked.

## 4. Run locally

```bash
python -m uvicorn app.main:app --reload --port 3000
```

To chat with the bot from your machine, use polling mode:

```env
APP_ENV=development
TELEGRAM_MODE=polling
```

The log line on startup tells you which transport is active:

```
Telegram transport=polling (local development)
```

> Only one consumer can receive Telegram updates at a time. If a webhook is
> registered, local polling gets 409 conflicts — delete the webhook first with
> `python scripts/set_telegram_webhook.py --delete`.

## 5. Try it without Telegram

```bash
curl http://localhost:3000/health
curl http://localhost:3000/mcp/tools
curl http://localhost:3000/test/gmail

curl -X POST http://localhost:3000/test/agent \
  -H "Content-Type: application/json" \
  -d '{"message":"what emails did I get today?"}'

curl -X POST http://localhost:3000/test/briefing
```

`/test/agent` keeps its own conversation history, so follow-up questions work.

## 6. Run the tests

```bash
python -m pytest -q
```

## 7. Environment variables

| Variable | Required | Notes |
| --- | --- | --- |
| `APP_ENV` | yes | `development` locally; `production` on Vercel |
| `TELEGRAM_BOT_TOKEN` | yes | From @BotFather |
| `TELEGRAM_CHAT_ID` | yes | Only this chat is served |
| `TELEGRAM_MODE` | no | `polling` or `webhook`; overrides the `APP_ENV` default |
| `TELEGRAM_WEBHOOK_SECRET` | production | Must match the value registered with Telegram. Not the bot token |
| `PUBLIC_BASE_URL` | production | e.g. `https://your-app.vercel.app` |
| `TIMEZONE` | yes | e.g. `Asia/Kolkata` |
| `BRIEFING_TIME` | yes | `HH:MM`, 24-hour |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | yes | Google Cloud OAuth client |
| `GOOGLE_REDIRECT_URI` | yes | Must match the environment you authorize from |
| `YOUTUBE_CHANNEL_IDS` | no | Comma-separated `UC...` channel IDs |
| `LLM_PROVIDER` | yes | `groq` (or `openai` / `gemini` / `none`) |
| `LLM_API_KEY` | yes | Provider key |
| `LLM_MODEL` | no | e.g. `llama-3.1-8b-instant` |
| `LLM_BASE_URL` | no | e.g. `https://api.groq.com/openai/v1` |

With `LLM_PROVIDER=none` the agent falls back to a deterministic keyword router
instead of conversational tool calling.

## 8. Deploy to Vercel

Serverless can't run a polling loop, so production uses webhooks. Any `APP_ENV`
other than `development`/`dev`/`local` defaults to webhook mode automatically.

1. Set every variable from section 7 in the Vercel project, with
   `APP_ENV=production` and a `TELEGRAM_WEBHOOK_SECRET`.
2. Deploy.
3. Verify the deployment is live and in the right mode:

```bash
curl https://your-app.vercel.app/
curl https://your-app.vercel.app/telegram/webhook/status
```

4. Register the webhook. The script reads the token and secret from `.env`, so
   they never land in your shell history:

```bash
python scripts/set_telegram_webhook.py
python scripts/set_telegram_webhook.py --info
```

Your local `.env` must use the **same** `TELEGRAM_WEBHOOK_SECRET` as Vercel, or
Telegram's deliveries get rejected with 403. The **same `TELEGRAM_BOT_TOKEN`**
must also be set on Vercel (replies are sent from there).

Set `TELEGRAM_MODE=webhook` locally too, so your dev server stops stealing updates.

### Google access on Vercel

Vercel's filesystem is ephemeral, so the OAuth token can't be saved there.
Instead, authenticate once locally and copy the token into an env var:

1. Locally, with `GOOGLE_REDIRECT_URI=http://localhost:3000/auth/google/callback`,
   run the server and visit `http://localhost:3000/auth/google` to connect.
2. Export the token JSON (local only; disabled in production):

```bash
curl http://localhost:3000/auth/google/token
```

3. Copy the `google_token_json` value into a Vercel env var named
   `GOOGLE_TOKEN_JSON`, then redeploy.

The deployment reads this token and refreshes access in memory per request. If
the refresh token is ever revoked, re-run steps 1–3.

## 9. Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Service banner |
| GET | `/health` | Health check |
| GET | `/telegram/webhook/status` | Transport status (no secrets) |
| POST | `/api/telegram/webhook` | Webhook, secret-token header |
| POST | `/api/telegram/webhook/{secret}` | Webhook, secret in path |
| GET | `/auth/google` | Start Google OAuth |
| GET | `/mcp/tools` | List registered MCP tools |
| POST | `/test/agent` | Chat with the agent over HTTP |
| POST | `/test/briefing` | Send the evening briefing now |
| POST | `/test/telegram` | Send a test Telegram message |
| GET | `/test/gmail` | Raw Gmail MCP output |

## 10. Troubleshooting

**Bot replies locally but not in production** — no webhook is registered, or your
local poller is draining the queue. Check `--info` and set `TELEGRAM_MODE=webhook`
locally.

**Webhook returns 403** — the secret in `.env` doesn't match Vercel's.

**Webhook returns 503** — `TELEGRAM_WEBHOOK_SECRET` is missing on the deployment.

**`redirect_uri_mismatch` during OAuth** — `GOOGLE_REDIRECT_URI` doesn't match the
Google Cloud console entry for the environment you're authorizing from.

**"Please reconnect your Google account"** — the token is missing a scope or was
revoked. Re-run `/auth/google`.

## 11. Known limitations

- Conversation history is in-memory and per-process, so it resets on restart and
  on every Vercel cold start. Durable memory needs Redis or Postgres.
- The evening briefing scheduler only runs in non-serverless mode; on Vercel use
  `/test/briefing` or an external cron.
- OAuth tokens are stored on local disk, which is ephemeral on Vercel.
