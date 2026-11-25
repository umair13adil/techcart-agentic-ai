"""Shared CLI for the MCP servers: stdio (default) or streamable HTTP."""
from __future__ import annotations

import argparse
import os

from mcp.server.transport_security import TransportSecuritySettings


def serve(server, default_port: int) -> None:
    ap = argparse.ArgumentParser(description=f"MCP server: {server.name}")
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=default_port)
    a = ap.parse_args()
    if not a.http:
        server.run("stdio")
        return
    # DNS-rebinding protection stays on; container service names are allow-listed explicitly
    allowed = [h.strip() for h in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    security = TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                         allowed_hosts=[f"127.0.0.1:{a.port}", f"localhost:{a.port}", *allowed])
    server.run("streamable-http", host=a.host, port=a.port, transport_security=security)
