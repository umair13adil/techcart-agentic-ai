"""Single entry point used by the API. Every reply comes from the LangGraph agent on Microsoft Foundry
(Azure OpenAI). There is no templated fallback: if the model isn't configured or fails, the API says so (503) and
the failure is logged - an honest outage beats a fake answer."""
from __future__ import annotations

import logging
import time

from .. import services
from ..config import settings
from .mcp_bridge import MCPHub

log = logging.getLogger(__name__)


class AssistantUnavailable(RuntimeError):
    pass


class AgentRuntime:
    def __init__(self, hub: MCPHub):
        self.hub = hub
        self.graph = None
        self.error: str | None = None
        if not settings.llm_enabled:
            self.error = "Microsoft Foundry is not configured (set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY)"
        else:
            try:
                from .graph import build_graph

                self.graph = build_graph(hub)
            except Exception as exc:
                self.error = f"Agent failed to start: {exc}"
                log.exception("LangGraph agent failed to start")

    @property
    def mode(self) -> str:
        return "llm" if self.graph is not None else "unavailable"

    def _cfg(self, session_id: str, customer_id: str | None) -> dict:
        return {"configurable": {"thread_id": session_id}, "tags": ["techcart"],
                "metadata": {"session_id": session_id, "customer_id": customer_id or "guest"}}

    def _require(self) -> None:
        if self.graph is None:
            raise AssistantUnavailable(self.error or "Assistant unavailable")

    async def _render(self, result: dict, session_id: str, cfg: dict) -> dict:
        ui = result.get("ui") or {}
        snapshot = await self.graph.aget_state(cfg)  # version-independent way to detect a pending interrupt
        pending = snapshot.values.get("checkout_summary") if "confirm_checkout" in (snapshot.next or ()) else None
        last = next((m for m in reversed(result.get("messages", [])) if m.type == "ai" and m.content), None)
        tools = [m.name for m in result.get("messages", []) if m.type == "tool"]
        return {"agent": "llm", "reply": last.content if last else "", "products": ui.get("products", []),
                "pending": pending, "cart": services.cart_view(session_id),
                "intent": (result.get("understanding") or {}).get("intent"), "route": result.get("route"),
                "tools": tools, "result": ui.get("result")}

    async def _run(self, session_id: str, customer_id: str | None, user_text: str, payload) -> dict:
        self._require()
        services.log_chat(session_id, customer_id, "user", user_text)
        cfg = self._cfg(session_id, customer_id)
        t0 = time.perf_counter()
        try:
            r = await self.graph.ainvoke(payload, cfg)
            out = await self._render(r, session_id, cfg)
        except Exception as exc:
            ms = int((time.perf_counter() - t0) * 1000)
            services.log_chat(session_id, customer_id, "assistant", "", latency_ms=ms, status=f"error: {exc}"[:200])
            log.exception("assistant turn failed")
            raise AssistantUnavailable("The assistant hit an error. Please try again in a moment.") from exc
        ms = int((time.perf_counter() - t0) * 1000)
        services.log_chat(session_id, customer_id, "assistant", out["reply"], out["intent"], out["route"],
                          out["tools"], ms)
        out["latency_ms"] = ms
        return out

    async def chat(self, session_id: str, customer_id: str | None, message: str) -> dict:
        self._require()  # before importing LangChain, so a missing setup reports 503 rather than crashing
        from .graph import initial_input

        return await self._run(session_id, customer_id, message, initial_input(session_id, customer_id, message))

    async def confirm(self, session_id: str, customer_id: str | None, approved: bool, otp: str | None) -> dict:
        self._require()
        from langgraph.types import Command

        state = await self.graph.aget_state(self._cfg(session_id, customer_id))
        if not state.next:
            return {"agent": "llm", "reply": "There's no checkout waiting for confirmation.", "pending": None,
                    "cart": services.cart_view(session_id)}
        label = "[confirmed checkout]" if approved else "[cancelled checkout]"
        return await self._run(session_id, customer_id, label, Command(resume={"approved": approved, "otp": otp}))
