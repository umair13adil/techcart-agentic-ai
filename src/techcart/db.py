"""SQLite persistence (stdlib only). Swap for Azure SQL / Postgres via the same SQL in production."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
  product_id TEXT PRIMARY KEY, sku TEXT, title TEXT, brand TEXT, category TEXT, subcategory TEXT,
  price REAL, cost REAL, rating REAL, rating_count INTEGER, specs_json TEXT, features TEXT,
  description TEXT, source TEXT, image_url TEXT
);
CREATE TABLE IF NOT EXISTS inventory (
  product_id TEXT PRIMARY KEY, stock_on_hand INTEGER, reserved INTEGER DEFAULT 0,
  reorder_point INTEGER, lead_time_days INTEGER, warehouse TEXT
);
-- real users only: created on first Microsoft Entra ID sign-in
CREATE TABLE IF NOT EXISTS customers (
  customer_id TEXT PRIMARY KEY, name TEXT, email TEXT, phone TEXT, city TEXT, segment TEXT,
  is_business INTEGER DEFAULT 0, created_at TEXT, marketing_consent INTEGER DEFAULT 0,
  entra_oid TEXT, tenant_id TEXT, last_login_at TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY, customer_id TEXT, created_at TEXT, expires_at TEXT
);
CREATE TABLE IF NOT EXISTS orders (
  order_id TEXT PRIMARY KEY, customer_id TEXT, created_at TEXT, status TEXT, channel TEXT,
  payment_method TEXT, subtotal REAL, discount REAL, vat REAL, shipping REAL, total REAL,
  ship_city TEXT, is_anomaly INTEGER DEFAULT 0, risk_score REAL, risk_reasons TEXT, risk_model TEXT
);
CREATE TABLE IF NOT EXISTS order_items (
  order_id TEXT, product_id TEXT, qty INTEGER, unit_price REAL, line_total REAL
);
CREATE TABLE IF NOT EXISTS suppliers (
  supplier_id TEXT PRIMARY KEY, name TEXT, country TEXT, brands TEXT, payment_terms TEXT, lead_time_days INTEGER
);
CREATE TABLE IF NOT EXISTS purchase_orders (
  po_id TEXT PRIMARY KEY, supplier_id TEXT, created_at TEXT, expected_at TEXT, status TEXT, total_cost REAL
);
CREATE TABLE IF NOT EXISTS po_lines (po_id TEXT, product_id TEXT, qty INTEGER, unit_cost REAL);
CREATE TABLE IF NOT EXISTS quotations (
  quote_id TEXT PRIMARY KEY, kind TEXT, party_id TEXT, created_at TEXT, valid_until TEXT, status TEXT,
  total REAL, notes TEXT
);
CREATE TABLE IF NOT EXISTS quote_lines (quote_id TEXT, product_id TEXT, qty INTEGER, unit_price REAL);
CREATE TABLE IF NOT EXISTS receipts (
  receipt_id TEXT PRIMARY KEY, order_id TEXT, issued_at TEXT, amount REAL, payment_method TEXT, txn_ref TEXT
);
-- AI analysis persisted per query (search box and chat): intent, confidence, tier, extracted entities
CREATE TABLE IF NOT EXISTS search_log (
  ts TEXT, session_id TEXT, query TEXT, intent TEXT, confidence REAL, used_llm INTEGER,
  source TEXT, customer_id TEXT, entities TEXT, results INTEGER
);
-- every assistant turn (PII-redacted): which agent handled it, tools called, latency, outcome
CREATE TABLE IF NOT EXISTS chat_log (
  ts TEXT, session_id TEXT, customer_id TEXT, role TEXT, content TEXT, intent TEXT, route TEXT,
  tools TEXT, latency_ms INTEGER, status TEXT
);
CREATE TABLE IF NOT EXISTS agent_audit (
  ts TEXT, session_id TEXT, actor TEXT, action TEXT, payload TEXT
);
CREATE INDEX IF NOT EXISTS ix_orders_created ON orders(created_at);
CREATE INDEX IF NOT EXISTS ix_items_order ON order_items(order_id);
CREATE INDEX IF NOT EXISTS ix_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS ix_search_ts ON search_log(ts);
CREATE INDEX IF NOT EXISTS ix_chat_ts ON chat_log(ts);
CREATE UNIQUE INDEX IF NOT EXISTS ux_customers_oid ON customers(entra_oid, tenant_id);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def session(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def rows(conn: sqlite3.Connection, sql: str, params: tuple | dict = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]
