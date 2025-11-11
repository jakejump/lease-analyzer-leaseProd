from __future__ import annotations

import os

# Application version surfaced via /healthz and /version
APP_VERSION: str = os.getenv("APP_VERSION", "0.1.0")


def get_allowed_origins() -> list[str]:
    """Return allowed CORS origins for the FastAPI app.

    Priority:
    - NEXT_PUBLIC_WEB_ORIGIN env var (single origin)
    - Default localhost dev origins
    """
    env_origin = os.getenv("NEXT_PUBLIC_WEB_ORIGIN")
    if env_origin:
        return [env_origin]
    # Local dev defaults
    return [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]

import os
from typing import List


APP_VERSION = os.getenv("APP_VERSION", "0.1.0")


def get_allowed_origins() -> List[str]:
    # Comma-separated env var; fallback to common local URLs and prod preview
    raw = os.getenv("ALLOWED_ORIGINS")
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://lease-analyzer-7og7.vercel.app",
    ]


