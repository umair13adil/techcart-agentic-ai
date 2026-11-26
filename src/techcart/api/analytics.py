"""Admin analytics: pure SQL over what real use has persisted (orders placed through checkout, searches and assistant
turns with their AI analysis, risk scores written at checkout). Nothing here is generated. In production these
queries would run against a warehouse (Fabric / Synapse) fed by CDC."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from .. import services
from ..config import settings


def _since(days: int, offset: int = 0) -> tuple[str, str]:
    end = datetime.now() - timedelta(days=offset)
    return (end - timedelta(days=days)).isoformat(), end.isoformat()


def kpis(days: int = 30) -> dict:
    def window(offset: int) -> dict:
        a, b = _since(days, offset)
        r = services.q("""SELECT COUNT(*) orders, COALESCE(SUM(total),0) revenue, COALESCE(AVG(total),0) aov,
                                 SUM(status='returned') returned, SUM(channel='assistant') via_assistant,
                                 SUM(risk_score >= 0.6) risky
                          FROM orders WHERE status!='cancelled' AND created_at>=? AND created_at<?""", (a, b))[0]
        units = services.q("""SELECT COALESCE(SUM(qty),0) u FROM order_items i JOIN orders o USING(order_id)
                              WHERE o.status!='cancelled' AND o.created_at>=? AND o.created_at<?""", (a, b))[0]["u"]
        margin = services.q("""SELECT COALESCE(SUM(i.line_total - i.qty*p.cost),0) m FROM order_items i
                               JOIN orders o USING(order_id) JOIN products p USING(product_id)
                               WHERE o.status!='cancelled' AND o.created_at>=? AND o.created_at<?""", (a, b))[0]["m"]
        return {**r, "units": units, "gross_margin": margin}

    cur, prev = window(0), window(days)
    pct = lambda a, b: round((a - b) / b * 100, 1) if b else None  # noqa: E731
    return {"days": days, "revenue": round(cur["revenue"]), "revenue_delta_pct": pct(cur["revenue"], prev["revenue"]),
            "orders": cur["orders"], "orders_delta_pct": pct(cur["orders"], prev["orders"]), "aov": round(cur["aov"]),
            "aov_delta_pct": pct(cur["aov"], prev["aov"]), "units": cur["units"],
            "gross_margin_pct": round(cur["gross_margin"] / (cur["revenue"] / 1.15) * 100, 1) if cur["revenue"] else 0,
            "return_rate_pct": round((cur["returned"] or 0) / max(cur["orders"], 1) * 100, 2),
            "assistant_share_pct": round((cur["via_assistant"] or 0) / max(cur["orders"], 1) * 100, 1),
            "risky_orders": cur["risky"] or 0}


def timeseries(days: int = 120) -> dict:
    a, _ = _since(days)
    r = services.q("""SELECT substr(created_at,1,10) d, COUNT(*) orders, ROUND(SUM(total)) revenue FROM orders
                      WHERE status!='cancelled' AND created_at>=? GROUP BY d ORDER BY d""", (a,))
    fc_path = settings.artifacts_dir / "forecast.json"
    fc = json.loads(fc_path.read_text()) if fc_path.exists() else []
    return {"history": r, "forecast": fc}


def by_category(days: int = 30) -> list[dict]:
    a, _ = _since(days)
    return services.q("""SELECT p.category, ROUND(SUM(i.line_total)) revenue, SUM(i.qty) units FROM order_items i
                         JOIN orders o USING(order_id) JOIN products p USING(product_id)
                         WHERE o.status!='cancelled' AND o.created_at>=? GROUP BY 1 ORDER BY 2 DESC""", (a,))


def top_products(days: int = 30, n: int = 10) -> list[dict]:
    a, _ = _since(days)
    return services.q("""SELECT p.product_id, p.title, p.category, SUM(i.qty) units, ROUND(SUM(i.line_total)) revenue
                         FROM order_items i JOIN orders o USING(order_id) JOIN products p USING(product_id)
                         WHERE o.status!='cancelled' AND o.created_at>=? GROUP BY 1 ORDER BY 5 DESC LIMIT ?""", (a, n))


def risky_orders(n: int = 12) -> list[dict]:
    rows = services.q("""SELECT o.order_id, o.created_at, o.total, o.payment_method, o.ship_city, o.risk_score,
                                o.risk_reasons, o.risk_model, c.city AS profile_city,
                                substr(c.created_at,1,10) AS customer_since
                         FROM orders o LEFT JOIN customers c USING(customer_id)
                         WHERE o.risk_score >= 0.6 ORDER BY o.created_at DESC LIMIT ?""", (n,))
    for r in rows:
        r["risk_reasons"] = json.loads(r["risk_reasons"] or "[]")
    return rows


def recent_orders(n: int = 15) -> list[dict]:
    rows = services.q("""SELECT o.order_id, o.created_at, o.status, o.total, o.payment_method, o.ship_city, o.channel,
                                o.risk_score, o.risk_reasons, c.name AS customer,
                                (SELECT COUNT(*) FROM order_items i WHERE i.order_id=o.order_id) AS lines
                         FROM orders o LEFT JOIN customers c USING(customer_id)
                         ORDER BY o.created_at DESC LIMIT ?""", (n,))
    for r in rows:
        r["risk_reasons"] = json.loads(r["risk_reasons"] or "[]")
    return rows


def data_status() -> dict:
    """How much real activity exists - drives the dashboard's empty states and the models' cold-start notes."""
    one = lambda sql: services.q(sql)[0]["n"]  # noqa: E731
    days = services.q("SELECT COUNT(DISTINCT substr(created_at,1,10)) n FROM orders WHERE status!='cancelled'")[0]["n"]
    return {"products": one("SELECT COUNT(*) n FROM products"), "customers": one("SELECT COUNT(*) n FROM customers"),
            "orders": one("SELECT COUNT(*) n FROM orders"), "days_with_sales": days,
            "searches": one("SELECT COUNT(*) n FROM search_log"),
            "assistant_turns": one("SELECT COUNT(*) n FROM chat_log WHERE role='assistant'"),
            "first_activity": services.q("SELECT MIN(ts) n FROM search_log")[0]["n"]}


def assistant_stats(days: int = 7) -> dict:
    a, _ = _since(days)
    s = services.q("""SELECT COUNT(*) turns, AVG(latency_ms) avg_ms, SUM(status!='ok') errors,
                             COUNT(DISTINCT session_id) sessions
                      FROM chat_log WHERE role='assistant' AND ts>=?""", (a,))[0]
    routes = services.q("""SELECT COALESCE(route,'-') route, COUNT(*) n FROM chat_log
                           WHERE role='assistant' AND ts>=? GROUP BY 1 ORDER BY 2 DESC""", (a,))
    tools: dict[str, int] = {}
    for r in services.q("SELECT tools FROM chat_log WHERE role='assistant' AND ts>=?", (a,)):
        for t in json.loads(r["tools"] or "[]"):
            tools[t] = tools.get(t, 0) + 1
    lat = sorted(r["latency_ms"] for r in services.q(
        "SELECT latency_ms FROM chat_log WHERE role='assistant' AND status='ok' AND ts>=?", (a,)) if r["latency_ms"])
    return {"turns": s["turns"], "sessions": s["sessions"], "errors": s["errors"] or 0,
            "p50_ms": lat[len(lat) // 2] if lat else None, "p95_ms": lat[int(len(lat) * 0.95)] if lat else None,
            "routes": routes, "tools": sorted(({"tool": k, "n": v} for k, v in tools.items()), key=lambda x: -x["n"]),
            "orders_via_assistant": services.q("SELECT COUNT(*) n FROM orders WHERE channel='assistant' AND created_at>=?",
                                               (a,))[0]["n"]}


def search_insights(days: int = 7) -> dict:
    a, _ = _since(days)
    intents = services.q("SELECT intent, COUNT(*) n FROM search_log WHERE ts>=? GROUP BY 1 ORDER BY 2 DESC", (a,))
    stats = services.q("SELECT COUNT(*) n, AVG(confidence) conf, AVG(used_llm) llm FROM search_log WHERE ts>=?", (a,))[0]
    top = services.q("""SELECT query, COUNT(*) n FROM search_log WHERE ts>=? AND intent='product_search'
                        GROUP BY 1 ORDER BY 2 DESC LIMIT 8""", (a,))
    low = services.q("""SELECT query, intent, ROUND(MIN(confidence),2) confidence, COUNT(*) n FROM search_log
                        WHERE ts>=? AND confidence<0.55 GROUP BY query ORDER BY n DESC, MAX(ts) DESC LIMIT 8""", (a,))
    cats: dict[str, int] = {}
    budgets = []
    for r in services.q("SELECT entities FROM search_log WHERE ts>=? AND entities IS NOT NULL", (a,)):
        e = json.loads(r["entities"] or "{}")
        if e.get("category"):
            cats[e["category"]] = cats.get(e["category"], 0) + 1
        if e.get("max_price"):
            budgets.append(e["max_price"])
    budgets.sort()
    return {"intents": intents, "queries": stats["n"], "avg_confidence": round(stats["conf"] or 0, 3),
            "llm_fallback_rate": round(stats["llm"] or 0, 3), "top_queries": top, "needs_labeling": low,
            "asked_categories": sorted(({"category": k, "n": v} for k, v in cats.items()), key=lambda x: -x["n"]),
            "median_budget": budgets[len(budgets) // 2] if budgets else None,
            "zero_result": services.q("""SELECT query, COUNT(*) n FROM search_log WHERE ts>=? AND results=0
                                         GROUP BY 1 ORDER BY 2 DESC LIMIT 6""", (a,))}


def audit_log(n: int = 25) -> list[dict]:
    return services.q("SELECT * FROM agent_audit ORDER BY ts DESC LIMIT ?", (n,))
