"""Single entry point used by the API: LLM graph when configured, rule agent otherwise or on LLM failure."""
from __future__ import annotations

import logging

from ..config import settings
from . import rule_agent
from .mcp_bridge import MCPHub

log = logging.getLogger(__name__)


class AgentRuntime:
    def __init__(self, hub: MCPHub):
        self.hub = hub
        self.graph = None
        self.mode = "rules"
        if settings.agent_mode == "llm" and settings.llm_enabled:
            try:
                from .graph import build_graph

                self.graph = build_graph(hub)
                self.mode = "llm"
            except Exception as exc:  # missing deps / config -> degrade, don't crash the shop
                log.warning("LLM agent unavailable (%s); using rule agent", exc)

    def _cfg(self, session_id: str, customer_id: str | None) -> dict:
        return {"configurable": {"thread_id": session_id}, "tags": ["techcart", self.mode],
                "metadata": {"session_id": session_id, "customer_id": customer_id or "guest"}}

    async def _render(self, result: dict, session_id: str, cfg: dict) -> dict:
        from .. import services

        ui = result.get("ui") or {}
        snapshot = await self.graph.aget_state(cfg)  # version-independent way to detect a pending interrupt
        pending = snapshot.values.get("checkout_summary") if "confirm_checkout" in (snapshot.next or ()) else None
        last = next((m for m in reversed(result.get("messages", [])) if m.type == "ai" and m.content), None)
        reply = last.content if last else ""
        if pending and not ui.get("result"):
            reply = (f"Ready to place your order: {pending['total']:,.0f} SAR via "
                     f"{pending['payment_method'].replace('_', ' ')} to {pending['city']}. Please confirm.")
        return {"agent": "llm", "reply": reply, "products": ui.get("products", []), "pending": pending,
                "cart": services.cart_view(session_id), "intent": (result.get("understanding") or {}).get("intent"),
                "route": result.get("route"), "result": ui.get("result")}

    async def chat(self, session_id: str, customer_id: str | None, message: str) -> dict:
        if self.graph is not None:
            try:
                from .graph import initial_input

                cfg = self._cfg(session_id, customer_id)
                r = await self.graph.ainvoke(initial_input(session_id, customer_id, message), cfg)
                return await self._render(r, session_id, cfg)
            except Exception as exc:
                log.exception("LLM agent failed, falling back to rules: %s", exc)
        return await rule_agent.handle(self.hub, session_id, customer_id, message)

    async def confirm(self, session_id: str, customer_id: str | None, approved: bool, otp: str | None) -> dict:
        if self.graph is not None:
            from langgraph.types import Command

            cfg = self._cfg(session_id, customer_id)
            state = await self.graph.aget_state(cfg)
            if state.next:  # graph is paused at the interrupt
                r = await self.graph.ainvoke(Command(resume={"approved": approved, "otp": otp}), cfg)
                return await self._render(r, session_id, cfg)
        return await rule_agent.confirm(self.hub, session_id, approved, otp)
