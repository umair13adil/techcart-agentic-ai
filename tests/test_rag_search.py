from techcart import services
from techcart.rag.evaluate import retrieval_metrics
from techcart.rag.retriever import answer, get_retriever
from techcart.search import search_products


def test_retrieval_quality_and_security():
    m = retrieval_metrics()
    assert m["hybrid"]["hit@5"] >= 0.85
    assert m["hybrid"]["mrr"] >= m["bm25"]["mrr"] - 0.02
    assert m["security_leaks"] == 0


def test_customer_cannot_see_internal_docs():
    for h in get_retriever().retrieve("purchase order PO supplier quotation total", role="customer", k=10):
        assert h.meta["audience"] == "public"


def test_identifier_lookup():
    qid = services.q("SELECT quote_id FROM quotations WHERE kind='customer_quote' LIMIT 1")[0]["quote_id"]
    hits = get_retriever().retrieve(f"status of {qid}?", role="staff", k=3)
    assert hits[0].doc_id == qid


def test_receipts_are_owner_scoped():
    r = services.q("SELECT receipt_id, order_id FROM receipts ORDER BY issued_at DESC LIMIT 1")[0]
    owner = services.q("SELECT customer_id FROM orders WHERE order_id=?", (r["order_id"],))[0]["customer_id"]
    q = f"receipt {r['receipt_id']}"
    assert all(h.doc_id != r["receipt_id"] for h in get_retriever().retrieve(q, role="customer", owner_id="C99999"))


def test_answer_extractive_fallback_cites():
    a = answer("How many days do I have to return an item?")
    assert a["citations"] and a["citations"][0]["doc_id"] == "policy-returns"


def test_search_respects_hard_filters():
    r = search_products("gaming laptop under 6000 SAR with 16GB RAM", k=8)
    assert r["results"] and not r["relaxed"]
    for p in r["results"]:
        assert p["category"] == "Laptops" and p["price"] <= 6000 and p["specs"]["ram_gb"] >= 16


def test_search_relaxes_impossible_constraints():
    r = search_products("RTX 4090 laptop under 1000 SAR", k=5)
    assert r["relaxed"]  # transparent relaxation instead of empty results
