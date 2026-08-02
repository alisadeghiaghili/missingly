# Production Security Checklist

Before exposing this project to the public internet:

1. Set `APP_ENV=production`.
2. Set a strong `SECRET_KEY` with at least 32 random characters.
3. Use `ADMIN_PASSWORD_HASH`; never use `ADMIN_PASSWORD` in production.
4. Set `ALLOWED_ORIGINS` to the exact production domain, for example `https://deepintelligence.ir`.
5. Put the app behind HTTPS with Nginx, Caddy, Cloudflare, or a managed platform.
6. Keep Uvicorn bound to `127.0.0.1` behind a reverse proxy.
7. Set `TRUST_PROXY_HEADERS=true` only when the reverse proxy is trusted and configured correctly.
8. Set `ZARINPAL_SANDBOX=false` only after real gateway credentials are configured.
9. Set `DATABASE_URL` to a managed PostgreSQL database; the app creates its schema at startup.
10. Set `REDIS_URL` to a managed Redis instance. Production deliberately refuses to start without it so rate limits cannot be bypassed by scaling workers.
11. Keep `.env`, `users.db`, logs, API keys, payment authorities, and reset tokens out of git, zip artifacts, and static paths.
12. Set `SESSION_TTL_DAYS` to the shortest practical duration for your product and periodically revoke inactive sessions.

Current hardening already applied:

- CORS is restricted by `ALLOWED_ORIGINS`.
- Bearer sessions and password-reset tokens are stored only as SHA-256 hashes; reset also revokes all active sessions.
- Admin rendering escapes server-provided values before inserting them into HTML.
- Admin bootstrap supports `ADMIN_PASSWORD_HASH`.
- Auth, password reset, admin APIs, payment tools, and code generation use Redis rate limits when `REDIS_URL` is configured; local development has an in-memory fallback.
- Generated preview HTML is sandboxed in an iframe.
- Basic security headers are added to every response.
- Zarinpal callback is idempotent for already verified payments.
- SQL inputs use parameter binding through helper functions, plus stricter length/format validation on auth, payment, reset, model, and code-generation inputs.
- Prompt-injection attempts that ask for hidden prompts, secrets, environment variables, API keys, or instruction override behavior are blocked before model calls.
- Model prompts explicitly mark user requests and existing HTML as untrusted content.

Recommended next hardening:

- CI installs pinned dependencies, runs tests, and produces a release ZIP that excludes local secrets, databases, logs, and virtual environments.
- Audit events are retained for admin changes, provider selection, projects, and successful payment verification.
- Replace the remaining inline-script CSP allowance with hashed external assets before a high-security deployment.
- Add a generated-HTML sanitizer/CSP allowlist if public users can publish generated pages.
- Migrate database access to PostgreSQL with a migration tool such as Alembic before production traffic.
