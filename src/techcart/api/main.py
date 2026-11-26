"""TechCart API (FastAPI). Serves the shop + admin dashboards and exposes the AI capabilities as REST endpoints
(OpenAPI docs at /docs) so backend/mobile teams can consume them like any other microservice.

    uvicorn techcart.api.main:app --reload --port 8000

Identity: customers and admins sign in with Microsoft Entra ID (/auth/login). The customer id used for orders,
recommendations and chat always comes from the server-side session - never from the request body.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import auth, services
from ..agents.mcp_bridge import MCPHub
from ..agents.runtime import AgentRuntime, AssistantUnavailable
from ..config import settings
from ..events import BUS, sse
from ..ml.drift import drift_report
from ..ml.tracking import Registry
from ..rag.retriever import answer
from ..search import search_products
from . import analytics

log = logging.getLogger("techcart")
WEB = Path(__file__).resolve().parents[1] / "web"
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub = await MCPHub().connect()
    STATE["hub"], STATE["runtime"] = hub, AgentRuntime(hub)
    log.info("TechCart up: assistant=%s sign-in=%s mcp=%s tools=%d", STATE["runtime"].mode,
             "entra" if auth.configured() else "NOT CONFIGURED", hub.mode, len(hub.tools))
    yield
    await hub.close()


app = FastAPI(title="TechCart AI Commerce API", version="0.2.0", lifespan=lifespan)


# ------------------------------------------------------------------------------------------ identity helpers
def current_user(request: Request) -> dict | None:
    return auth.user_from_token(request.cookies.get(auth.COOKIE))


def require_user(request: Request) -> dict:
    if not auth.configured():
        raise HTTPException(503, "Sign-in is not configured (set ENTRA_CLIENT_ID / ENTRA_CLIENT_SECRET).")
    u = current_user(request)
    if not u:
        raise HTTPException(401, "Please sign in with your Microsoft account.")
    return u


def require_admin(request: Request) -> dict:
    u = require_user(request)
    if not u["is_admin"]:
        raise HTTPException(403, "Your account is not an administrator (ADMIN_EMAILS).")
    return u


# ------------------------------------------------------------------------------------------ pages + auth
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


@app.get("/auth/login", include_in_schema=False)
async def login(next: str = "/"):
    if not auth.configured():
        return HTMLResponse("<h3>Microsoft sign-in is not configured.</h3><p>Set ENTRA_CLIENT_ID and "
                            "ENTRA_CLIENT_SECRET in .env (see README), then restart the API.</p>", status_code=503)
    return RedirectResponse(auth.begin_login(next), status_code=302)


@app.get("/auth/callback", include_in_schema=False)
async def auth_callback(request: Request):
    try:
        token, next_path = await asyncio.to_thread(auth.complete_login, dict(request.query_params))
    except PermissionError as exc:
        return HTMLResponse(f"<h3>Sign-in failed</h3><p>{exc}</p><p><a href='/auth/login'>Try again</a></p>",
                            status_code=400)
    resp = RedirectResponse(next_path, status_code=302)
    resp.set_cookie(auth.COOKIE, token, max_age=settings.session_ttl_hours * 3600, httponly=True, samesite="lax",
                    secure=request.url.scheme == "https", path="/")
    return resp


@app.get("/auth/logout", include_in_schema=False)
async def logout(request: Request):
    auth.end_session(request.cookies.get(auth.COOKIE))
    home = str(request.base_url)
    resp = RedirectResponse(auth.logout_url(home) if auth.configured() else home, status_code=302)
    resp.delete_cookie(auth.COOKIE, path="/")
    return resp


# ------------------------------------------------------------------------------------------ system + profile
@app.get("/api/health")
async def health():
    return {"ok": True, "time": datetime.now().isoformat(timespec="seconds")}


@app.get("/api/system")
async def system():
    rt: AgentRuntime = STATE["runtime"]
    return {"assistant": rt.mode, "assistant_error": rt.error, "llm_configured": settings.llm_enabled,
            "auth_configured": auth.configured(), "mcp_mode": rt.hub.mode, "currency": settings.currency}


@app.get("/api/me")
async def me(request: Request):
    return {"user": current_user(request), "auth_configured": auth.configured()}


class ProfileIn(BaseModel):
    city: str = Field(min_length=2, max_length=60)
    phone: str | None = Field(None, max_length=20)
    marketing_consent: bool = False


@app.post("/api/me/profile")
async def update_profile(body: ProfileIn, request: Request):
    u = require_user(request)
    auth.update_profile(u["customer_id"], body.city, body.phone, body.marketing_consent)
    return {"user": current_user(request)}


# ------------------------------------------------------------------------------------------ shop
class ChatIn(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=2000)


class ConfirmIn(BaseModel):
    session_id: str
    approved: bool
    otp: str | None = None


class CartIn(BaseModel):
    session_id: str
    product_id: str
    qty: int = 1


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/api/session")
async def new_session():
    return {"session_id": services.new_session()}


@app.get("/api/home")
async def home(request: Request, session_id: str | None = None):
    from ..agents.tools import recommend_for_customer

    u = current_user(request)
    cid = u["customer_id"] if u else None
    has_history = bool(cid and services.customer_history(cid))
    recs = recommend_for_customer(cid, session_id, limit=8)
    store_sellers = analytics.top_products(14, 6)
    shown = {r["product_id"] for r in recs}  # never show the same product in both rows
    trending = [services.product_card(services.product(p["product_id"])) for p in store_sellers] or \
        services.list_products(sort="popular", limit=6 + len(shown))["items"]
    trending = [t for t in trending if t["product_id"] not in shown][:6]
    return {"user": u, "personalised": bool(u and u["marketing_consent"] and has_history),
            "recommendations": recs, "recent_orders": services.recent_orders(cid, 3) if cid else [],
            "trending": trending, "trending_source": "store_sales" if store_sellers else "amazon_reviews"}


@app.get("/api/search")
async def search(request: Request, q: str, session_id: str = "anon", k: int = 12):
    r = await asyncio.to_thread(search_products, q, k)
    u, me_ = r["understanding"], current_user(request)
    services.log_search(session_id, q, u["intent"], u["confidence"], u["tier"] == "llm", source="search",
                        customer_id=me_["customer_id"] if me_ else None, entities=u["entities"],
                        results=len(r["results"]))
    return r


@app.post("/api/chat")
async def chat(body: ChatIn, request: Request):
    u = current_user(request)
    try:
        return await STATE["runtime"].chat(body.session_id, u["customer_id"] if u else None, body.message)
    except AssistantUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/chat/confirm")
async def chat_confirm(body: ConfirmIn, request: Request):
    u = require_user(request)  # money moves only for a signed-in customer
    try:
        return await STATE["runtime"].confirm(body.session_id, u["customer_id"], body.approved, body.otp)
    except AssistantUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


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


# ------------------------------------------------------------------------------------------ admin (Entra + allow-list)
@app.get("/api/admin/overview")
async def admin_overview(request: Request):
    """Everything the dashboard shows, computed only from what real use has persisted."""
    require_admin(request)
    return {"data": analytics.data_status(), "models": Registry().info()}


@app.get("/api/admin/kpis")
async def admin_kpis(request: Request, days: int = 30):
    require_admin(request)
    return analytics.kpis(days)


@app.get("/api/admin/timeseries")
async def admin_ts(request: Request, days: int = 120):
    require_admin(request)
    return analytics.timeseries(days)


@app.get("/api/admin/categories")
async def admin_categories(request: Request, days: int = 30):
    require_admin(request)
    return analytics.by_category(days)


@app.get("/api/admin/top-products")
async def admin_top(request: Request, days: int = 30):
    require_admin(request)
    return analytics.top_products(days)


@app.get("/api/admin/risky-orders")
async def admin_risky(request: Request):
    require_admin(request)
    return analytics.risky_orders()


@app.get("/api/admin/recent-orders")
async def admin_recent_orders(request: Request):
    require_admin(request)
    return analytics.recent_orders()


@app.get("/api/admin/low-stock")
async def admin_low_stock(request: Request):
    """Served through the ERP MCP server - the dashboard consumes the same integration the agents use."""
    require_admin(request)
    return await STATE["hub"].call("low_stock_report", limit=12)


@app.get("/api/admin/search-insights")
async def admin_search(request: Request):
    require_admin(request)
    return analytics.search_insights()


@app.get("/api/admin/assistant")
async def admin_assistant(request: Request):
    require_admin(request)
    return analytics.assistant_stats()


@app.get("/api/admin/drift")
async def admin_drift(request: Request):
    require_admin(request)
    return await asyncio.to_thread(drift_report, services.conn())


@app.get("/api/admin/audit")
async def admin_audit(request: Request):
    require_admin(request)
    return analytics.audit_log()


@app.get("/api/admin/system")
async def admin_system(request: Request):
    require_admin(request)
    rt: AgentRuntime = STATE["runtime"]
    return {**(await system()), "mcp_tools": rt.hub.describe()}


@app.post("/api/admin/ask")
async def admin_ask(body: AskIn, request: Request):
    """Ops copilot: RAG over policies, products and business documents (staff scope)."""
    require_admin(request)
    return await asyncio.to_thread(answer, body.question, "staff")


@app.get("/api/admin/stream")
async def admin_stream(request: Request):
    require_admin(request)
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
