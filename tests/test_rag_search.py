from techcart import services
from techcart.agents.tools import request_b2b_quote
from techcart.rag.evaluate import retrieval_metrics
from techcart.rag.retriever import answer, build_index, get_retriever
from techcart.search import search_products


def test_retrieval_quality():
    m = retrieval_metrics()
    assert m["hybrid"]["hit@5"] >= 0.85
    assert m["hybrid"]["mrr"] >= m["bm25"]["mrr"] - 0.02
    assert m["security_leaks"] == 0


def test_real_business_documents_are_indexed_with_security_trimming():
    """A quotation created through the app becomes searchable for staff after re-indexing - never for customers."""
    services.execute("INSERT INTO customers (customer_id, name, email, created_at) VALUES ('UTESTB2B', 'Acme', "
                     "'buyer@acme.example', '2026-01-01T00:00:00')")
    laptop = next(p for p in services.all_products() if p["category"] == "Laptops")
    qt = request_b2b_quote("UTESTB2B", laptop["product_id"], 25)
    build_index("hashing")
    get_retriever.cache_clear()
    staff = get_retriever().retrieve(f"status of quotation {qt['quote_id']}", role="staff", k=3)
    assert staff[0].doc_id == qt["quote_id"]
    customer = get_retriever().retrieve(f"quotation {qt['quote_id']}", role="customer", k=10)
    assert all(h.meta["audience"] == "public" for h in customer)


def test_answer_extractive_fallback_cites():
    a = answer("How many days do I have to return an item?")
    assert a["citations"] and a["citations"][0]["doc_id"] == "policy-returns"


def test_search_respects_hard_filters():
    r = search_products("laptop under 6000 SAR with 16GB RAM", k=8)
    assert r["results"]
    for p in r["results"]:
        assert p["category"] == "Laptops" and p["price"] <= 6000 * (1.2 if "price" in r["relaxed"] else 1)


def test_search_relaxes_impossible_constraints():
    r = search_products("RTX 4090 laptop under 1000 SAR", k=5)
    assert r["relaxed"]  # transparent relaxation instead of empty results
