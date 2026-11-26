"""TechCart API (FastAPI). Serves the shop + admin dashboards and exposes the AI capabilities as REST endpoints
(OpenAPI docs at /docs) so backend/mobile teams can consume them like any other microservice.

    uvicorn techcart.api.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import services
from ..agents.mcp_bridge import MCPHub
from ..agents.runtime import AgentRuntime
from ..config import settings
from ..ml.drift import drift_report
from ..ml.tracking import Registry
from ..rag.retriever import answer
from ..search import search_products
from . import analytics
from .simulator import BUS, run_simulator, sse

log = logging.getLogger("techcart")
WEB = Path(__file__).resolve().parents[1] / "web"
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub = await MCPHub().connect()
    STATE["hub"], STATE["runtime"] = hub, AgentRuntime(hub)
    task = asyncio.create_task(run_simulator()) if os.environ.get("SIMULATE", "1") == "1" else None
    log.info("TechCart up: agent=%s mcp=%s tools=%d", STATE["runtime"].mode, hub.mode, len(hub.tools))
    yield
    if task:
        task.cancel()
    await hub.close()


app = FastAPI(title="TechCart AI Commerce API", version="0.1.0", lifespan=lifespan)


# ------------------------------------------------------------------------------------------ pages
@app.get("/", include_in_schema=False)
async def shop_page():
    return FileResponse(WEB / "shop.html")


@app.get("/admin", include_in_schema=False)
async def admin_page():
    return FileResponse(WEB / "admin.html")


@app.get("/web/{name}", include_in_schema=False)
async def web_asset(name: str):
    if name not in {"charts.js"}:  # explicit allow-list, no path traversal
        raise HTTPException(404)
    return FileResponse(WEB / name, media_type="application/javascript")


# ------------------------------------------------------------------------------------------ system
@app.get("/api/health")
async def health():
    return {"ok": True, "time": datetime.now().isoformat(timespec="seconds")}


@app.get("/api/system")
async def system():
    rt: AgentRuntime = STATE["runtime"]
    return {"agent_mode": rt.mode, "llm_configured": settings.llm_enabled, "mcp_mode": rt.hub.mode,
            "mcp_tools": rt.hub.describe(), "models": Registry().info(), "currency": settings.currency}


# ------------------------------------------------------------------------------------------ customer
class ChatIn(BaseModel):
    session_id: str
    customer_id: str | None = None
    message: str = Field(min_length=1, max_length=2000)


class ConfirmIn(BaseModel):
    session_id: str
    customer_id: str | None = None
    approved: bool
    otp: str | None = None


class CartIn(BaseModel):
    session_id: str
    product_id: str
    qty: int = 1


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/api/customers/demo")
async def demo_customers():
    """A few realistic customers with history (for the 'sign in as' switcher in the demo shop)."""
    return services.q("""SELECT c.customer_id, c.name, c.city, c.segment, c.marketing_consent, COUNT(o.order_id) orders
                         FROM customers c JOIN orders o USING(customer_id) WHERE c.is_business=0
                         GROUP BY c.customer_id HAVING orders BETWEEN 3 AND 8 AND c.marketing_consent=1
                         ORDER BY c.segment, orders DESC LIMIT 40""")[::5][:6]


@app.get("/api/session")
async def new_session():
    return {"session_id": services.new_session()}


@app.get("/api/home")
async def home(customer_id: str | None = None, session_id: str | None = None):
    from ..agents.tools import recommend_for_customer

    cust = services.customer(customer_id) if customer_id else None
    recs = recommend_for_customer(customer_id, session_id, limit=8)
    personalised = bool(cust and cust["marketing_consent"] and services.customer_history(customer_id))
    return {"customer": cust, "personalised": personalised, "recommendations": recs,
            "recent_orders": services.recent_orders(customer_id, 3) if customer_id else [],
            "trending": [services.product_card(services.product(p["product_id"])) for p in analytics.top_products(14, 6)]}


@app.get("/api/search")
async def search(q: str, session_id: str = "anon", k: int = 12):
    r = await asyncio.to_thread(search_products, q, k)
    u = r["understanding"]
    services.log_search(session_id, q, u["intent"], u["confidence"], u["tier"] == "llm")
    return r


@app.post("/api/chat")
async def chat(body: ChatIn):
    return await STATE["runtime"].chat(body.session_id, body.customer_id, body.message)


@app.post("/api/chat/confirm")
async def chat_confirm(body: ConfirmIn):
    return await STATE["runtime"].confirm(body.session_id, body.customer_id, body.approved, body.otp)


@app.post("/api/cart/add")
async def cart_add(body: CartIn):
    try:
        return services.cart_add(body.session_id, body.product_id, body.qty)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/cart/{session_id}")
async def cart(session_id: str):
    return services.cart_view(session_id)


@app.get("/api/categories")
async def categories():
    return {"total": len(services.all_products()), "categories": services.catalog_counts()}


@app.get("/api/products")
async def products(category: str | None = None, subcategory: str | None = None, sort: str = "popular",
                   offset: int = 0, limit: int = 12):
    limit = max(1, min(limit, 60))
    return services.list_products(category, subcategory, sort, max(0, offset), limit)


@app.get("/api/products/{product_id}")
async def product(product_id: str):
    p = services.product(product_id)
    if not p:
        raise HTTPException(404, "not found")
    return {**services.product_card(p), "description": p["description"], "features": p["features"].split("\n")}


@app.post("/api/ask")
async def ask(body: AskIn):
    return await asyncio.to_thread(answer, body.question, "customer")


# ------------------------------------------------------------------------------------------ admin
@app.get("/api/admin/kpis")
async def admin_kpis(days: int = 30):
    return analytics.kpis(days)


@app.get("/api/admin/timeseries")
async def admin_ts(days: int = 120):
    return analytics.timeseries(days)


@app.get("/api/admin/categories")
async def admin_categories(days: int = 30):
    return analytics.by_category(days)


@app.get("/api/admin/top-products")
async def admin_top(days: int = 30):
    return analytics.top_products(days)


@app.get("/api/admin/risky-orders")
async def admin_risky():
    return analytics.risky_orders()


@app.get("/api/admin/low-stock")
async def admin_low_stock():
    """Served through the ERP MCP server - the dashboard consumes the same integration the agents use."""
    return await STATE["hub"].call("low_stock_report", limit=12)


@app.get("/api/admin/search-insights")
async def admin_search():
    return analytics.search_insights()


@app.get("/api/admin/drift")
async def admin_drift():
    return await asyncio.to_thread(drift_report, services.conn())


@app.get("/api/admin/audit")
async def admin_audit():
    return analytics.audit_log()


@app.post("/api/admin/ask")
async def admin_ask(body: AskIn):
    """Ops copilot: RAG over purchase orders, quotations and policies (internal audience)."""
    return await asyncio.to_thread(answer, body.question, "staff")


@app.get("/api/admin/stream")
async def admin_stream(request: Request):
    q = BUS.subscribe()

    async def gen():
        try:
            yield sse({"type": "hello", "ts": datetime.now().isoformat(timespec="seconds")})
            while not await request.is_disconnected():
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield sse(ev)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            BUS.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
