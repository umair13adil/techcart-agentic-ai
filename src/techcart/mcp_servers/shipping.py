"""MCP server: Logistics (mock carrier aggregator - SMSA / Aramex / own fleet).

    python -m techcart.mcp_servers.shipping --http --port 8103
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from mcp.server.mcpserver import MCPServer

server = MCPServer("logistics", instructions="Shipping quotes, shipment creation and tracking within Saudi Arabia.")
SHIPMENTS: dict[str, dict] = {}
ZONES = {"riyadh": 1, "jeddah": 2, "dammam": 2, "khobar": 2, "mecca": 2, "medina": 2}


@server.tool(description="Quote shipping options for a KSA city. order_value in SAR (free standard above 200).")
def quote_shipping(city: str, order_value: float, weight_kg: float = 2.0) -> list[dict]:
    zone = ZONES.get(city.strip().lower(), 3)
    std_fee = 0.0 if order_value >= 200 else 25.0
    opts = [{"service": "standard", "carrier": "TechCart Fleet" if zone == 1 else "SMSA Express",
             "fee": std_fee, "eta_days": {1: 1, 2: 3, 3: 5}[zone]}]
    if zone == 1:
        opts.append({"service": "same_day", "carrier": "TechCart Fleet", "fee": 49.0, "eta_days": 0,
                     "cutoff": "12:00"})
    if weight_kg > 10:
        opts = [{**o, "carrier": "Aramex Freight", "note": "white-glove"} for o in opts]
    return opts


@server.tool(description="Create a shipment for a paid order; returns a tracking number.")
def create_shipment(order_id: str, city: str, service: str = "standard") -> dict:
    trk = f"TC{random.randint(10**9, 10**10 - 1)}SA"
    zone = ZONES.get(city.strip().lower(), 3)
    eta = datetime.now() + timedelta(days=0 if service == "same_day" else {1: 1, 2: 3, 3: 5}[zone])
    SHIPMENTS[trk] = {"tracking_number": trk, "order_id": order_id, "city": city, "service": service,
                      "eta": eta.date().isoformat(), "created": datetime.now().isoformat(timespec="seconds")}
    return SHIPMENTS[trk]


@server.tool(description="Tracking events for a tracking number.")
def track(tracking_number: str) -> dict:
    s = SHIPMENTS.get(tracking_number)
    if not s:
        return {"error": "unknown tracking number"}
    created = datetime.fromisoformat(s["created"])
    events = [{"ts": created.isoformat(timespec="minutes"), "event": "Label created"},
              {"ts": (created + timedelta(hours=3)).isoformat(timespec="minutes"), "event": "Picked up from RUH-1"}]
    return {**s, "events": events, "status": "in_transit"}


def main() -> None:
    from ._run import serve

    serve(server, 8103)


if __name__ == "__main__":
    main()
