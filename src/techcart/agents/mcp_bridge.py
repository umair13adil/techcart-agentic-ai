"""Minimal MCP client hub (no adapter library): connects to several MCP servers, lists their tools, calls them,
and can wrap them as LangChain tools for the LLM agents.

Written directly against the MCP Python SDK v2 `Client`, which accepts an in-process server object (tests/demo),
a URL (streamable HTTP, e.g. separate containers) or stdio parameters - one code path for all three.
"""
from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any

from mcp.client.client import Client

from ..config import settings


def _servers(mode: str) -> dict[str, Any]:
    if mode == "http":
        return dict(settings.mcp_urls)
    from ..mcp_servers import inventory_erp, payments, shipping

    return {"inventory": inventory_erp.server, "payments": payments.server, "shipping": shipping.server}


class MCPHub:
    def __init__(self, mode: str | None = None):
        self.mode = mode or settings.mcp_mode
        self.clients: dict[str, Client] = {}
        self.tools: dict[str, tuple[str, Any]] = {}  # tool name -> (server, Tool)
        self._stack: AsyncExitStack | None = None
        self.calls: list[dict] = []  # trace of MCP calls (shown in the UI "under the hood" panel)

    async def connect(self) -> "MCPHub":
        self._stack = AsyncExitStack()
        for name, target in _servers(self.mode).items():
            client = await self._stack.enter_async_context(Client(target))
            self.clients[name] = client
            for t in (await client.list_tools()).tools:
                self.tools[t.name] = (name, t)
        return self

    async def close(self) -> None:
        if self._stack:
            await self._stack.aclose()

    async def call(self, tool: str, **args) -> Any:
        server, _ = self.tools[tool]
        res = await self.clients[server].call_tool(tool, args)
        texts = [getattr(c, "text", "") for c in res.content]
        text = "".join(texts)
        sc = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
        if isinstance(sc, dict):
            out = sc["result"] if set(sc) == {"result"} else sc   # wrapped list/primitive results
        else:
            try:
                out = json.loads(text) if len(texts) == 1 else [json.loads(t) for t in texts]
            except json.JSONDecodeError:
                out = text
        self.calls.append({"server": server, "tool": tool, "args": args, "ok": not res.is_error})
        if res.is_error:
            raise RuntimeError(f"MCP {server}.{tool} failed: {text}")
        return out

    def describe(self) -> list[dict]:
        return [{"server": s, "tool": n, "description": t.description} for n, (s, t) in self.tools.items()]

    def as_langchain_tools(self, include: set[str] | None = None) -> list:
        from langchain_core.tools import StructuredTool

        out = []
        for name, (server, t) in self.tools.items():
            if include and name not in include:
                continue
            schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {"type": "object"}

            async def _run(_name=name, **kwargs):
                return json.dumps(await self.call(_name, **kwargs), default=str)

            out.append(StructuredTool.from_function(coroutine=_run, name=name,
                                                    description=f"[{server} via MCP] {t.description}",
                                                    args_schema=schema))
        return out
