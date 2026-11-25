"""MCP server: the TechCart store itself, exposed OUTWARDS so external agents (Claude Desktop, Copilot,
a partner's procurement agent) can search the catalog, read policies and get recommendations.

Only read-only, public-audience capabilities are exposed - no checkout, no customer PII.

Claude Desktop config (claude_desktop_config.json):
    {"mcpServers": {"techcart": {"command": "python", "args": ["-m", "techcart.mcp_servers.techcart_store"],
                                 "cwd": "/path/to/techcart", "env": {"PYTHONPATH": "src"}}}}
"""
from __future__ import annotations


from mcp.server.mcpserver import MCPServer

from .. import services
from ..data.catalog import CATEGORIES

server = MCPServer("techcart-store", instructions="Search TechCart's computer & accessories catalog (Saudi Arabia, "
                                                  "prices in SAR incl. VAT) and answer policy questions with citations.")


@server.tool(description="Natural-language product search, e.g. 'gaming laptop under 5000 SAR with 16GB RAM'.")
def search_products(query: str, limit: int = 8) -> dict:
    from ..search import search_products as _search

    r = _search(query, k=limit)
    return {"matched_on": r["matched_on"], "relaxed": r["relaxed"],
            "results": [{k: x[k] for k in ("product_id", "title", "price", "rating", "stock_label")} for x in r["results"]]}


@server.tool(description="Full details and specs for one product id.")
def get_product(product_id: str) -> dict:
    p = services.product(product_id)
    return services.product_card(p) if p else {"error": "not found"}


@server.tool(description="Products frequently bought together with the given product ids.")
def frequently_bought_with(product_ids: list[str], limit: int = 6) -> list[dict]:
    from datetime import datetime

    from ..ml.tracking import Registry

    rec = Registry().load("recommender")
    hist = [(p, datetime.now()) for p in product_ids]
    out = rec.recommend(hist, k=limit, in_stock={k for k, v in services.stock_map().items() if v > 0},
                        now=datetime.now(), explain=True)
    return [{**services.product_card(services.product(r["product_id"])), "reason": r.get("reason")} for r in out]


@server.tool(description="Answer a question about TechCart policies (returns, warranty, shipping, payments, B2B, "
                         "privacy) with citations.")
def ask_policy(question: str) -> dict:
    from ..rag.retriever import answer

    return answer(question, role="customer")


@server.resource("techcart://categories", description="Product categories", mime_type="text/plain")
def categories() -> str:
    return "\n".join(CATEGORIES)


@server.prompt(description="System prompt for a shopping assistant using this server")
def shopping_assistant() -> str:
    return ("You help customers choose computers and accessories from TechCart. Use search_products to find items, "
            "get_product for specs, frequently_bought_with for accessories and ask_policy for policies. Quote prices "
            "in SAR and always mention stock status.")


def main() -> None:
    from ._run import serve

    serve(server, 8104)


if __name__ == "__main__":
    main()
