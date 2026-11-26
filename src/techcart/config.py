"""Central settings. Reads environment variables (and a local .env if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #")[0].strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":  # KEY="value" / KEY='value'
            v = v[1:-1]
        os.environ.setdefault(k.strip(), v)


_load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("TECHCART_DATA_DIR", str(ROOT / "data"))).resolve())
    currency: str = field(default_factory=lambda: _env("CURRENCY", "SAR"))
    mcp_mode: str = field(default_factory=lambda: _env("MCP_MODE", "inproc"))
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "local"))

    aoai_endpoint: str = field(default_factory=lambda: _env("AZURE_OPENAI_ENDPOINT"))
    aoai_key: str = field(default_factory=lambda: _env("AZURE_OPENAI_API_KEY"))
    aoai_api_version: str = field(default_factory=lambda: _env("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"))
    aoai_chat_deployment: str = field(default_factory=lambda: _env("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5.4-mini"))
    aoai_embed_deployment: str = field(
        default_factory=lambda: _env("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")
    )

    # Microsoft Entra ID sign-in
    entra_tenant: str = field(default_factory=lambda: _env("ENTRA_TENANT", "common"))
    entra_client_id: str = field(default_factory=lambda: _env("ENTRA_CLIENT_ID"))
    entra_client_secret: str = field(default_factory=lambda: _env("ENTRA_CLIENT_SECRET"))
    entra_redirect_uri: str = field(default_factory=lambda: _env("ENTRA_REDIRECT_URI", "http://localhost:8000/auth/callback"))
    admin_emails: frozenset = field(default_factory=lambda: frozenset(
        e.strip().lower() for e in _env("ADMIN_EMAILS").split(",") if e.strip()))
    session_ttl_hours: int = field(default_factory=lambda: int(_env("SESSION_TTL_HOURS", "12")))

    mcp_urls: dict = field(
        default_factory=lambda: {
            "inventory": _env("MCP_INVENTORY_URL", "http://localhost:8101/mcp"),
            "payments": _env("MCP_PAYMENTS_URL", "http://localhost:8102/mcp"),
            "shipping": _env("MCP_SHIPPING_URL", "http://localhost:8103/mcp"),
        }
    )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "techcart.db"

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "docs"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.aoai_endpoint and self.aoai_key)


settings = Settings()
