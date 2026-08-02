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
9. Move from local SQLite to PostgreSQL before public production traffic.
10. Keep `.env`, `users.db`, logs, API keys, payment authorities, and reset tokens out of git and out of static paths.

Current hardening already applied:

- CORS is restricted by `ALLOWED_ORIGINS`.
- Admin bootstrap supports `ADMIN_PASSWORD_HASH`.
- Auth, password reset, admin APIs, payment tools, and code generation have in-memory rate limits.
- Generated preview HTML is sandboxed in an iframe.
- Basic security headers are added to every response.
- Zarinpal callback is idempotent for already verified payments.
- SQL inputs use parameter binding through helper functions, plus stricter length/format validation on auth, payment, reset, model, and code-generation inputs.
- Prompt-injection attempts that ask for hidden prompts, secrets, environment variables, API keys, or instruction override behavior are blocked before model calls.
- Model prompts explicitly mark user requests and existing HTML as untrusted content.

Recommended next hardening:

- Add persistent/distributed rate limiting such as Redis before running multiple app instances.
- Add audit logging for every admin plan edit and payment action.
- Add a generated-HTML sanitizer/CSP allowlist if public users can publish generated pages.
- Migrate database access to PostgreSQL with a migration tool such as Alembic before production traffic.
