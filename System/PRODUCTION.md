# Production Readiness Guide

This document describes how to run this project with production-safe defaults.

## 1) Required environment variables

Set these before start:

- `APP_ENV=production`
- `SECRET_KEY=<strong-random-secret>`
- `HOST=0.0.0.0`
- `PORT=5000`

Recommended:

- `DEBUG=false`
- `ALLOW_UNSAFE_WERKZEUG=false`
- `CORS_ALLOWED_ORIGINS=https://your-domain.com`
- `MAX_CONTENT_LENGTH=1048576`
- `RATE_LIMIT_ENABLED=true`
- `RATE_LIMIT_PER_MINUTE=180`
- `ENABLE_DEMO_AUTOLOGIN=false`
- `ENABLE_NETWORK_SNIFFING=false` (unless explicitly required)
- `ENABLE_ENDPOINT_SIMULATION=false`

## 2) MongoDB

Set a reachable MongoDB instance (default is `mongodb://localhost:27017/`).

If MongoDB is unavailable, the app falls back to in-memory storage (not suitable for production persistence).

## 3) Start command

Use your environment Python:

```powershell
& "C:/Users/mihir/Downloads/Data sets/System/System/.venv/Scripts/python.exe" app.py
```

## 4) Security notes

- In production, `SECRET_KEY` is mandatory.
- Demo auto-login is disabled by default when `APP_ENV=production`.
- Session cookies are set to secure defaults (`HttpOnly`, `SameSite=Lax`, `Secure=true` in production).
- Basic per-IP rate limiting is available for incoming requests.

## 5) Operational health check

- `GET /healthz`

Returns app status and current environment.

## 6) API Analyzer in production

- Start API monitoring from the UI.
- Real incoming `/api...` HTTP requests are captured and analyzed live.
- Internal API analyzer control endpoints and static/socket paths are excluded from capture.

## 7) Remaining hardening recommendations

For enterprise deployment, still add:

- Reverse proxy / WAF in front of app
- Proper authN/authZ for sensitive routes
- Centralized logs and metrics
- Async queue for high-volume inference
- CI/CD checks for model/schema compatibility
