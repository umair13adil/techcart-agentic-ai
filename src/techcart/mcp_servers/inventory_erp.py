"""MCP server: TechCart ERP (inventory, reservations, suppliers, purchase orders).

Represents the "other system" a real store would integrate with (SAP / Oracle / Odoo / Dynamics). The agent never
touches the ERP database directly; it calls these tools over MCP, so the same integration works for LangGraph,
Claude Desktop, Copilot Studio or any MCP client.

    python -m techcart.mcp_servers.inventory_erp                          # stdio
    python -m techcart.mcp_servers.inventory_erp --http --port 8101       # streamable HTTP
"""
from __future__ import annotations

import json
import threading
from datetime import date, timedelta

from mcp.server.mcpserver import MCPServer

from ..db import connect, rows

server = MCPServer("techcart-erp", instructions="Inventory, stock reservations, suppliers and purchase orders "
                                                 "for TechCart. Reservations are idempotent per order_ref.")
_lock = threading.Lock()
_conn = None
RESERVATIONS: dict[str, list[dict]] = {}  # order_ref -> [{product_id, qty}]


def _db():
    global _conn
    if _conn is None:
        _conn = connect()
    return _conn


@server.tool(description="Available stock (on hand minus reserved) for up to 20 product ids.")
def check_stock(product_ids: list[str]) -> dict:
    ph = ",".join("?" * len(product_ids[:20]))
    r = rows(_db(), f"SELECT product_id, stock_on_hand, reserved, warehouse, lead_time_days FROM inventory "
                    f"WHERE product_id IN ({ph})", tuple(product_ids[:20]))
    return {x["product_id"]: {"available": x["stock_on_hand"] - x["reserved"], "warehouse": x["warehouse"],
                              "restock_days": x["lead_time_days"]} for x in r}


@server.tool(description="Reserve stock for an order. Idempotent: calling again with the same order_ref returns "
                         "the existing reservation. Fails atomically if any line is short.")
def reserve_stock(order_ref: str, lines: list[dict]) -> dict:
    with _lock:
        if order_ref in RESERVATIONS:
            return {"order_ref": order_ref, "status": "reserved", "lines": RESERVATIONS[order_ref], "idempotent": True}
        avail = check_stock([l["product_id"] for l in lines])
        short = [l for l in lines if avail.get(l["product_id"], {}).get("available", 0) < int(l["qty"])]
        if short:
            return {"order_ref": order_ref, "status": "insufficient_stock",
                    "short": [{"product_id": l["product_id"], "requested": l["qty"],
                               "available": avail.get(l["product_id"], {}).get("available", 0)} for l in short]}
        for l in lines:
            _db().execute("UPDATE inventory SET reserved = reserved + ? WHERE product_id=?", (int(l["qty"]), l["product_id"]))
        _db().commit()
        RESERVATIONS[order_ref] = [{"product_id": l["product_id"], "qty": int(l["qty"])} for l in lines]
        return {"order_ref": order_ref, "status": "reserved", "lines": RESERVATIONS[order_ref], "ttl_minutes": 15}


@server.tool(description="Release a reservation (payment failed / customer cancelled).")
def release_reservation(order_ref: str) -> dict:
    with _lock:
        lines = RESERVATIONS.pop(order_ref, [])
        for l in lines:
            _db().execute("UPDATE inventory SET reserved = MAX(0, reserved - ?) WHERE product_id=?", (l["qty"], l["product_id"]))
        _db().commit()
    return {"order_ref": order_ref, "released": len(lines)}


@server.tool(description="Convert a reservation into a stock deduction after successful payment.")
def commit_reservation(order_ref: str) -> dict:
    with _lock:
        lines = RESERVATIONS.pop(order_ref, None)
        if lines is None:
            return {"order_ref": order_ref, "status": "not_found"}
        for l in lines:
            _db().execute("UPDATE inventory SET reserved = MAX(0, reserved - ?), stock_on_hand = stock_on_hand - ? "
                          "WHERE product_id=?", (l["qty"], l["qty"], l["product_id"]))
        _db().commit()
    return {"order_ref": order_ref, "status": "committed", "lines": lines}


@server.tool(description="Products at or below their reorder point, with suggested reorder quantity and supplier.")
def low_stock_report(limit: int = 15) -> list[dict]:
    return rows(_db(), """SELECT i.product_id, p.title, p.brand, i.stock_on_hand - i.reserved AS available,
                                 i.reorder_point, MAX(i.reorder_point * 2 - (i.stock_on_hand - i.reserved), 1) AS suggested_qty,
                                 i.lead_time_days
                          FROM inventory i JOIN products p USING(product_id)
                          WHERE i.stock_on_hand - i.reserved <= i.reorder_point
                          ORDER BY (i.stock_on_hand - i.reserved) * 1.0 / MAX(i.reorder_point, 1) LIMIT ?""", (limit,))


@server.tool(description="Create a DRAFT purchase order to a supplier. Drafts require human approval in the ERP "
                         "before they are sent.")
def create_purchase_order(supplier_id: str, lines: list[dict], note: str = "") -> dict:
    sup = rows(_db(), "SELECT * FROM suppliers WHERE supplier_id=?", (supplier_id,))
    if not sup:
        return {"error": f"unknown supplier {supplier_id}"}
    n = rows(_db(), "SELECT COUNT(*) n FROM purchase_orders")[0]["n"] + 1
    po_id = f"PO-{date.today().year}-{n:05d}D"
    total = 0.0
    for l in lines:
        cost = rows(_db(), "SELECT cost FROM products WHERE product_id=?", (l["product_id"],))
        unit = cost[0]["cost"] if cost else 0
        total += unit * int(l["qty"])
        _db().execute("INSERT INTO po_lines VALUES (?,?,?,?)", (po_id, l["product_id"], int(l["qty"]), unit))
    _db().execute("INSERT INTO purchase_orders VALUES (?,?,?,?,?,?)",
                  (po_id, supplier_id, date.today().isoformat(),
                   (date.today() + timedelta(days=sup[0]["lead_time_days"])).isoformat(), "draft", round(total, 2)))
    _db().commit()
    return {"po_id": po_id, "status": "draft", "total_cost": round(total, 2), "requires_approval": True, "note": note}


@server.tool(description="Fetch a purchase order with its lines.")
def get_purchase_order(po_id: str) -> dict:
    po = rows(_db(), "SELECT * FROM purchase_orders WHERE po_id=?", (po_id,))
    if not po:
        return {"error": "not found"}
    return {**po[0], "lines": rows(_db(), "SELECT * FROM po_lines WHERE po_id=?", (po_id,))}


@server.resource("erp://suppliers", description="Supplier master data", mime_type="application/json")
def suppliers() -> str:
    return json.dumps(rows(_db(), "SELECT * FROM suppliers"))


def main() -> None:
    from ._run import serve

    serve(server, 8101)


if __name__ == "__main__":
    main()
