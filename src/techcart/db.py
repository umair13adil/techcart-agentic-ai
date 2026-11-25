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
CREATE TABLE IF NOT EXISTS customers (
  customer_id TEXT PRIMARY KEY, name TEXT, email TEXT, phone TEXT, city TEXT, segment TEXT,
  is_business INTEGER, created_at TEXT, marketing_consent INTEGER
);
CREATE TABLE IF NOT EXISTS orders (
  order_id TEXT PRIMARY KEY, customer_id TEXT, created_at TEXT, status TEXT, channel TEXT,
  payment_method TEXT, subtotal REAL, discount REAL, vat REAL, shipping REAL, total REAL,
  ship_city TEXT, is_anomaly INTEGER DEFAULT 0, risk_score REAL
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
CREATE TABLE IF NOT EXISTS search_log (
  ts TEXT, session_id TEXT, query TEXT, intent TEXT, confidence REAL, used_llm INTEGER
);
CREATE TABLE IF NOT EXISTS agent_audit (
  ts TEXT, session_id TEXT, actor TEXT, action TEXT, payload TEXT
);
CREATE INDEX IF NOT EXISTS ix_orders_created ON orders(created_at);
CREATE INDEX IF NOT EXISTS ix_items_order ON order_items(order_id);
CREATE INDEX IF NOT EXISTS ix_orders_customer ON orders(customer_id);
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
