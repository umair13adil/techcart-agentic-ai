"""Synthetic-but-realistic transactional data for TechCart.

Generates customers, 18 months of orders (with seasonality, personas, complementary baskets and
repeat "next purchase" behaviour), receipts, suppliers, purchase orders, supplier & customer quotations,
inventory, and the RAG document corpus + a labelled RAG evaluation set.

Everything is seeded => reproducible. Injected anomalies are labelled (orders.is_anomaly) so the fraud
model can be evaluated honestly.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from .catalog import COMPLEMENTS, DURABLE
from .policies import POLICIES

CITIES = [("Riyadh", 0.38), ("Jeddah", 0.22), ("Dammam", 0.1), ("Khobar", 0.06), ("Mecca", 0.07),
          ("Medina", 0.06), ("Abha", 0.04), ("Tabuk", 0.03), ("Buraydah", 0.04)]
FIRST = ["Mohammed", "Abdullah", "Fahad", "Khalid", "Omar", "Faisal", "Saad", "Turki", "Nasser", "Yousef",
         "Ahmed", "Ali", "Sultan", "Majed", "Hassan", "Nora", "Sara", "Reem", "Lama", "Hessa", "Aisha", "Fatimah",
         "Maha", "Noura", "Dana", "Jawaher", "Rana", "Layla", "Umair", "Ravi", "John", "Maria"]
LAST = ["Al-Qahtani", "Al-Otaibi", "Al-Ghamdi", "Al-Harbi", "Al-Zahrani", "Al-Dosari", "Al-Shehri", "Al-Mutairi",
        "Al-Anazi", "Al-Shammari", "Al-Malki", "Al-Subaie", "Khan", "Siddiqui", "Hussain", "Fernandes", "Thomas"]
COMPANIES = ["Najd Engineering Co.", "Red Sea Logistics", "Al-Faisal Academy", "Riyadh Smart Clinics",
             "Eastern Petro Services", "Hijaz Tourism Group", "Saudi FinServe", "Qassim University Labs",
             "Tabuk Construction", "Gulf Insurance Partners", "Makkah Medical Center", "Vision Retail Co."]
PERSONAS = {  # weight, {category: (prob, preferred subcategories)}
    "student": (0.30, {"Laptops": (0.55, {"student", "ultrabook", "2-in-1"}), "Mice": (0.1, {"office"}),
                       "Bags & Sleeves": (0.1, None), "Headsets": (0.08, {"office"}), "Storage": (0.1, None),
                       "Chargers & Power": (0.07, None)}),
    "gamer": (0.25, {"Laptops": (0.32, {"gaming"}), "Desktops": (0.08, {"gaming desktop"}),
                     "Graphics Cards": (0.15, None), "Monitors": (0.15, {"gaming"}), "Mice": (0.12, {"gaming"}),
                     "Headsets": (0.1, {"gaming"}), "Keyboards": (0.08, {"mechanical"})}),
    "professional": (0.25, {"Laptops": (0.4, {"business", "ultrabook"}), "Monitors": (0.18, {"office", "creator"}),
                            "Docks & Hubs": (0.14, None), "Headsets": (0.1, {"office"}),
                            "Webcams": (0.08, {"office"}), "Keyboards": (0.05, {"office"}),
                            "Networking": (0.05, None)}),
    "creator": (0.12, {"Laptops": (0.4, {"creator", "ultrabook"}), "Monitors": (0.2, {"creator"}),
                       "Storage": (0.2, {"external ssd", "internal ssd"}), "Memory": (0.1, None),
                       "Desktops": (0.1, {"mini pc", "all-in-one"})}),
    "business": (0.08, {"Laptops": (0.55, {"business"}), "Monitors": (0.2, {"office"}),
                        "Docks & Hubs": (0.15, {"dock"}), "Headsets": (0.1, {"office"})}),
}
PAYMENT = [("mada", 0.35), ("credit_card", 0.25), ("apple_pay", 0.2), ("tabby", 0.12), ("cod", 0.08)]
SUPPLIERS = [
    ("SUP01", "Gulf IT Distribution LLC", "UAE", ["Dell", "HP", "Lenovo", "Microsoft"], "Net 45", 7),
    ("SUP02", "Arabian Tech Distributors", "KSA", ["ASUS", "Acer", "MSI", "Gigabyte", "Sapphire"], "Net 30", 5),
    ("SUP03", "Peninsula Apple Partners", "KSA", ["Apple"], "Prepaid", 4),
    ("SUP04", "Desert Peripherals Trading", "KSA", ["Logitech", "Razer", "SteelSeries", "HyperX", "Keychron",
                                                    "Jabra", "Sony", "Elgato", "Microsoft"], "Net 30", 6),
    ("SUP05", "Al-Wadi Storage & Memory", "UAE", ["Samsung", "WD", "Crucial", "SanDisk", "Seagate", "Kingston",
                                                  "Corsair", "G.Skill"], "Net 30", 8),
    ("SUP06", "Kingdom Accessories Co.", "KSA", ["Anker", "UGREEN", "Targus", "Samsonite", "tomtoc", "Incase",
                                                  "TP-Link", "APC", "Cooler Master", "KLIM", "CalDigit", "BenQ",
                                                  "LG", "Intel"], "Net 15", 5),
]


def _w(choices, rng):
    r, acc = rng.random() * sum(w for _, w in choices), 0.0
    for c, w in choices:
        acc += w
        if r <= acc:
            return c
    return choices[-1][0]


def _seasonality(d: date) -> float:
    m = 1.0
    if d.month == 11 and d.day >= 13:            # White Friday season (KSA's Black Friday)
        m *= 2.4 if 24 <= d.day <= 30 else 1.6
    if (d.month == 8 and d.day >= 15) or (d.month == 9 and d.day <= 15):
        m *= 1.45                                 # back to school / university
    ramadan = {2025: (date(2025, 3, 1), date(2025, 3, 30)), 2026: (date(2026, 2, 18), date(2026, 3, 19))}
    for s, e in ramadan.values():
        if s <= d <= e:
            m *= 1.25
    if d.month == 12 and d.day >= 15:
        m *= 1.15                                 # year-end budgets
    m *= {3: 1.15, 4: 1.25, 5: 0.9}.get(d.weekday(), 1.0)  # Thu/Fri peak, Sat dip (KSA weekend Fri-Sat)
    return m


class Generator:
    def __init__(self, products: list[dict], seed: int = 42, days: int = 540, end: date | None = None,
                 n_customers: int = 3200, base_daily: float = 20.0):
        self.rng = random.Random(seed)
        self.products = products
        self.by_cat: dict[str, list[dict]] = defaultdict(list)
        for p in products:
            self.by_cat[p["category"]].append(p)
        self.end = end or date.today()
        self.start = self.end - timedelta(days=days)
        self.n_customers = n_customers
        self.base_daily = base_daily
        self.customers: list[dict] = []
        self.orders: list[dict] = []
        self.items: list[dict] = []

    # ------------------------------------------------------------------ helpers
    def _pick_product(self, cat: str, subs: set | None, brand_hint: str | None = None) -> dict | None:
        pool = self.by_cat.get(cat) or []
        if not pool:
            return None
        if subs:
            pool = [p for p in pool if p["subcategory"] in subs] or pool
        if brand_hint and self.rng.random() < 0.55:
            branded = [p for p in pool if p["brand"] == brand_hint]
            pool = branded or pool
        # popularity: higher rating_count -> more likely; cheaper items slightly more likely
        weights = [math.log1p(p["rating_count"]) + 1 / (1 + p["price"] / 2000) for p in pool]
        return self.rng.choices(pool, weights=weights, k=1)[0]

    def _persona_primary(self, persona: str, history_cats: list[str]) -> tuple[str, set | None]:
        prefs = PERSONAS[persona][1]
        # repeat customers often come back for a complement of a durable they already own
        durable_owned = [c for c in history_cats if c in DURABLE]
        if durable_owned and self.rng.random() < 0.6:
            comp = self.rng.choice(COMPLEMENTS[self.rng.choice(durable_owned)])
            return comp, prefs.get(comp, (0, None))[1]
        cat = _w([(c, v[0]) for c, v in prefs.items()], self.rng)
        return cat, prefs[cat][1]

    # ------------------------------------------------------------------ customers
    def make_customers(self) -> None:
        for i in range(1, self.n_customers + 1):
            persona = _w([(k, v[0]) for k, v in PERSONAS.items()], self.rng)
            is_biz = persona == "business"
            name = self.rng.choice(COMPANIES) if is_biz else f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}"
            created = self.start - timedelta(days=200) + timedelta(days=int(self.rng.random() ** 0.8 * 700))
            self.customers.append({
                "customer_id": f"C{i:05d}", "name": name,
                "email": f"{name.split()[0].lower().replace('.', '')}{i}@example.com",
                "phone": f"+9665{self.rng.randint(0, 99_999_999):08d}", "city": _w(CITIES, self.rng),
                "segment": persona, "is_business": int(is_biz), "created_at": created.isoformat(),
                "marketing_consent": int(self.rng.random() < 0.62),
                "_loyalty": self.rng.lognormvariate(0, 1.0) * (2.5 if is_biz else 1),
            })

    # ------------------------------------------------------------------ orders
    def make_orders(self) -> None:
        hist: dict[str, list[str]] = defaultdict(list)
        brand_pref: dict[str, str] = {}
        seq = 0
        total_days = (self.end - self.start).days
        for day in range(total_days + 1):
            d = self.start + timedelta(days=day)
            trend = 0.75 + 0.5 * day / total_days  # business growing ~65% over the period
            n = max(0, int(self.rng.gauss(self.base_daily * trend * _seasonality(d), 4)))
            eligible = [c for c in self.customers if c["created_at"] <= d.isoformat()]
            if not eligible:
                continue
            weights = [c["_loyalty"] for c in eligible]
            for cust in self.rng.choices(eligible, weights=weights, k=n):
                seq += 1
                self._make_order(seq, d, cust, hist, brand_pref)
        # labelled anomalies (~0.8%) for the fraud model
        for _ in range(int(len(self.orders) * 0.008)):
            seq += 1
            d = self.start + timedelta(days=self.rng.randint(30, total_days))
            self._make_anomaly(seq, d)

    def _order_ts(self, d: date, odd: bool = False) -> datetime:
        hour = self.rng.choice([2, 3, 4]) if odd else int(min(23, max(8, self.rng.gauss(20, 3.5))))
        ts = datetime(d.year, d.month, d.day, hour, self.rng.randint(0, 59), self.rng.randint(0, 59))
        now = datetime.now()
        if ts > now:  # today's orders can't be in the future
            ts = (now - timedelta(minutes=self.rng.randint(1, max(2, now.hour * 60 + now.minute - 1)))).replace(microsecond=0)
        return ts

    def _finalise(self, oid, cust, ts, lines, city=None, payment=None, anomaly=0):
        subtotal = sum(l["line_total"] for l in lines)
        discount = 0.0
        if ts.month == 11 and ts.day >= 13:
            discount = round(subtotal * self.rng.choice([0.05, 0.1, 0.15]), 2)
        elif self.rng.random() < 0.12:
            discount = round(min(subtotal * 0.1, self.rng.choice([50, 100, 150])), 2)
        net = subtotal - discount
        shipping = 0.0 if net >= 200 else 25.0
        total = round(net + shipping, 2)
        vat = round(total * 15 / 115, 2)  # prices are VAT-inclusive
        age = (self.end - ts.date()).days
        if age <= 1:
            status = "processing"
        elif age <= 4:
            status = self.rng.choice(["shipped", "shipped", "delivered"])
        else:
            status = _w([("delivered", 0.92), ("returned", 0.04), ("cancelled", 0.04)], self.rng)
        pay = payment or _w(PAYMENT, self.rng)
        if pay == "cod" and total > 2500:
            pay = "mada"
        self.orders.append({
            "order_id": oid, "customer_id": cust["customer_id"], "created_at": ts.isoformat(), "status": status,
            "channel": _w([("web", 0.45), ("app", 0.45), ("assistant", 0.1)], self.rng), "payment_method": pay,
            "subtotal": round(subtotal, 2), "discount": discount, "vat": vat, "shipping": shipping, "total": total,
            "ship_city": city or cust["city"], "is_anomaly": anomaly, "risk_score": None,
        })
        for l in lines:
            self.items.append({"order_id": oid, **l})

    def _line(self, p: dict, qty: int = 1) -> dict:
        return {"product_id": p["product_id"], "qty": qty, "unit_price": p["price"],
                "line_total": round(p["price"] * qty, 2)}

    def _make_order(self, seq, d, cust, hist, brand_pref):
        persona = cust["segment"]
        cat, subs = self._persona_primary(persona, hist[cust["customer_id"]])
        primary = self._pick_product(cat, subs, brand_pref.get(cust["customer_id"]))
        if not primary:
            return
        qty = 1
        if cust["is_business"] and cat in ("Laptops", "Monitors", "Docks & Hubs", "Headsets"):
            qty = self.rng.choice([2, 3, 5, 5, 10, 12, 20, 25])
        lines = [self._line(primary, qty)]
        chosen = {primary["product_id"]}
        # complementary attach (decaying probability), brand-aware
        for i, comp in enumerate(self.rng.sample(COMPLEMENTS.get(cat, []), k=len(COMPLEMENTS.get(cat, [])))):
            if self.rng.random() < 0.42 * (0.62 ** i):
                psubs = PERSONAS[persona][1].get(comp, (0, None))[1]
                p = self._pick_product(comp, psubs, primary["brand"] if primary["brand"] == "Apple" else None)
                if comp == "Bags & Sleeves" and p:
                    screen = json.loads(primary["specs_json"]).get("screen_in", 14)
                    fit = [b for b in self.by_cat[comp] if json.loads(b["specs_json"]).get("fits_in", 99) >= screen]
                    p = self.rng.choice(fit) if fit else p
                if p and p["product_id"] not in chosen:
                    chosen.add(p["product_id"])
                    lines.append(self._line(p, qty if cust["is_business"] and comp == "Docks & Hubs" else 1))
        hist[cust["customer_id"]].extend({self._cat(l["product_id"]) for l in lines})
        if primary["category"] == "Laptops":
            brand_pref[cust["customer_id"]] = primary["brand"]
        self._finalise(f"O{seq:07d}", cust, self._order_ts(d), lines)

    def _make_anomaly(self, seq, d):
        """Fraud-like orders. Each signal appears only with some probability so that the pattern is not
        trivially separable (real fraudsters mimic normal behaviour)."""
        rng = self.rng
        base = rng.choice(self.customers)
        if rng.random() < 0.7:  # brand-new account
            cust = dict(base, customer_id=f"C9{seq:06d}", created_at=d.isoformat(), segment="student",
                        is_business=0, marketing_consent=0, _loyalty=0.0,
                        email=f"user{rng.randint(1000, 9999)}@mail-temp.example")
            self.customers.append(cust)
        else:  # account takeover of an existing customer
            cust = base if base["created_at"] <= d.isoformat() else self.customers[0]
        cat = rng.choice(["Graphics Cards", "Laptops", "Laptops", "Monitors"])
        expensive = sorted(self.by_cat[cat], key=lambda p: -p["price"])[:8]
        lines = [self._line(rng.choice(expensive), rng.choice([1, 2, 3, 4, 6, 8]))]
        city = rng.choice([c for c, _ in CITIES if c != cust["city"]]) if rng.random() < 0.6 else cust["city"]
        pay = "credit_card" if rng.random() < 0.8 else "apple_pay"
        self._finalise(f"O{seq:07d}", cust, self._order_ts(d, odd=rng.random() < 0.6), lines, city=city,
                       payment=pay, anomaly=1)

    def _cat(self, pid: str) -> str:
        if not hasattr(self, "_cat_idx"):
            self._cat_idx = {p["product_id"]: p["category"] for p in self.products}
        return self._cat_idx[pid]

    # ------------------------------------------------------------------ supply side
    def make_supply(self) -> dict:
        rng = self.rng
        pmap = {p["product_id"]: p for p in self.products}
        units_30, units_all = defaultdict(int), defaultdict(int)
        cutoff = (self.end - timedelta(days=30)).isoformat()
        o_date = {o["order_id"]: o["created_at"] for o in self.orders}
        for it in self.items:
            units_all[it["product_id"]] += it["qty"]
            if o_date[it["order_id"]] >= cutoff:
                units_30[it["product_id"]] += it["qty"]
        sup_for_brand = {b: s[0] for s in SUPPLIERS for b in s[3]}
        suppliers = [{"supplier_id": s[0], "name": s[1], "country": s[2], "brands": ",".join(s[3]),
                      "payment_terms": s[4], "lead_time_days": s[5]} for s in SUPPLIERS]
        lead = {s[0]: s[5] for s in SUPPLIERS}
        inventory = []
        for p in self.products:
            daily = units_30[p["product_id"]] / 30
            sid = sup_for_brand.get(p["brand"], "SUP06")
            rop = max(2, math.ceil(daily * (lead[sid] + 7)))
            soh = max(0, int(rop * rng.choice([0.2, 0.6, 1.2, 1.8, 2.5, 3.0]) + rng.randint(0, 6)))
            inventory.append({"product_id": p["product_id"], "stock_on_hand": soh, "reserved": 0,
                              "reorder_point": rop, "lead_time_days": lead[sid],
                              "warehouse": rng.choice(["RUH-1", "RUH-1", "JED-1", "DMM-1"])})
        # purchase orders: weekly per supplier over the period
        pos, po_lines = [], []
        n = 0
        by_sup = defaultdict(list)
        for p in self.products:
            by_sup[sup_for_brand.get(p["brand"], "SUP06")].append(p)
        d = self.start
        while d <= self.end:
            for sid, plist in by_sup.items():
                if rng.random() < 0.55:
                    continue
                n += 1
                pid = f"PO-{d.year}-{n:05d}"
                picks = rng.sample(plist, k=min(len(plist), rng.randint(2, 7)))
                total = 0.0
                for p in picks:
                    q = max(2, int(units_all[p["product_id"]] / 75 * rng.uniform(0.7, 1.6)))
                    po_lines.append({"po_id": pid, "product_id": p["product_id"], "qty": q, "unit_cost": p["cost"]})
                    total += q * p["cost"]
                age = (self.end - d).days
                status = "open" if age < 3 else "in_transit" if age < lead[sid] + 3 else \
                    _w([("received", 0.95), ("partially_received", 0.03), ("cancelled", 0.02)], rng)
                pos.append({"po_id": pid, "supplier_id": sid, "created_at": d.isoformat(),
                            "expected_at": (d + timedelta(days=lead[sid])).isoformat(), "status": status,
                            "total_cost": round(total, 2)})
            d += timedelta(days=7)
        # quotations: supplier RFQ responses + B2B customer quotes
        quotes, q_lines = [], []
        biz = [c for c in self.customers if c["is_business"]]
        for i in range(1, 221):
            d = self.start + timedelta(days=rng.randint(0, (self.end - self.start).days))
            if i % 2:  # customer quote
                c = rng.choice(biz)
                qid = f"QT-C-{i:04d}"
                laptops = [p for p in self.by_cat["Laptops"] if p["subcategory"] in ("business", "ultrabook")]
                lap = rng.choice(laptops)
                qty = rng.choice([10, 12, 15, 20, 25, 30, 40, 50, 60])
                disc = 0.12 if qty >= 50 else 0.08 if qty >= 25 else 0.05
                lines = [(lap, qty, round(lap["price"] * (1 - disc), 2))]
                if rng.random() < 0.7:
                    dock = rng.choice(self.by_cat["Docks & Hubs"])
                    lines.append((dock, qty, round(dock["price"] * (1 - disc), 2)))
                if rng.random() < 0.4:
                    mon = rng.choice([p for p in self.by_cat["Monitors"] if p["subcategory"] == "office"])
                    lines.append((mon, qty, round(mon["price"] * (1 - disc), 2)))
                age = (self.end - d).days
                status = "sent" if age <= 14 else _w([("accepted", 0.45), ("expired", 0.4), ("rejected", 0.15)], rng)
                party, kind, notes = c["customer_id"], "customer_quote", \
                    f"Volume discount {int(disc * 100)}% applied. Includes Autopilot enrolment option at 25 SAR/unit."
            else:  # supplier quote in response to our RFQ
                sid = rng.choice(list(by_sup))
                qid = f"QT-S-{i:04d}"
                picks = rng.sample(by_sup[sid], k=min(3, len(by_sup[sid])))
                lines = [(p, rng.choice([10, 20, 50, 100]), round(p["cost"] * rng.uniform(0.93, 1.04), 2))
                         for p in picks]
                status = _w([("accepted", 0.5), ("expired", 0.3), ("under_review", 0.2)], rng)
                party, kind = sid, "supplier_quote"
                notes = f"Prices ex-works, {dict((s[0], s[4]) for s in SUPPLIERS)[sid]} terms, delivery in {lead[sid]} days."
            total = round(sum(q * u for _, q, u in lines), 2)
            quotes.append({"quote_id": qid, "kind": kind, "party_id": party, "created_at": d.isoformat(),
                           "valid_until": (d + timedelta(days=14)).isoformat(), "status": status,
                           "total": total, "notes": notes})
            q_lines += [{"quote_id": qid, "product_id": p["product_id"], "qty": q, "unit_price": u}
                        for p, q, u in lines]
        receipts = []
        for o in self.orders:
            if o["status"] in ("cancelled", "processing") or (o["payment_method"] == "cod" and o["status"] != "delivered"):
                continue
            receipts.append({"receipt_id": "R" + o["order_id"][1:], "order_id": o["order_id"],
                             "issued_at": o["created_at"], "amount": o["total"],
                             "payment_method": o["payment_method"],
                             "txn_ref": f"TXN{rng.randint(10**9, 10**10 - 1)}"})
        return {"suppliers": suppliers, "inventory": inventory, "purchase_orders": pos, "po_lines": po_lines,
                "quotations": quotes, "quote_lines": q_lines, "receipts": receipts, "_pmap": pmap}


# ---------------------------------------------------------------------- persistence + documents
def write_db(conn, gen: Generator, supply: dict) -> None:
    from ..db import init_schema

    for t in ["products", "inventory", "customers", "orders", "order_items", "suppliers", "purchase_orders",
              "po_lines", "quotations", "quote_lines", "receipts", "search_log", "agent_audit"]:
        conn.execute(f"DROP TABLE IF EXISTS {t}")
    init_schema(conn)

    def ins(table, recs):
        if not recs:
            return
        cols = [k for k in recs[0] if not k.startswith("_")]
        conn.executemany(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                         [tuple(r[c] for c in cols) for r in recs])

    ins("products", gen.products)
    ins("customers", gen.customers)
    ins("orders", gen.orders)
    ins("order_items", gen.items)
    for k in ["suppliers", "inventory", "purchase_orders", "po_lines", "quotations", "quote_lines", "receipts"]:
        ins(k, supply[k])
    conn.commit()


def _money(x: float) -> str:
    return f"{x:,.2f} SAR"


def write_documents(out_dir: Path, gen: Generator, supply: dict, max_receipts: int = 250,
                    max_pos: int = 150) -> list[dict]:
    """Render business documents to Markdown + a manifest.jsonl with metadata for RAG/RBAC."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pmap = supply["_pmap"]
    cust = {c["customer_id"]: c for c in gen.customers}
    sup = {s["supplier_id"]: s for s in supply["suppliers"]}
    manifest: list[dict] = []

    def emit(doc_id, doc_type, audience, title, body, **meta):
        path = out_dir / doc_type / f"{doc_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {title}\n\n{body.strip()}\n")
        manifest.append({"doc_id": doc_id, "doc_type": doc_type, "audience": audience, "title": title,
                         "path": str(path.relative_to(out_dir)), **meta})

    for p in gen.products:
        specs = json.loads(p["specs_json"])
        spec_md = "\n".join(f"- **{k.replace('_', ' ')}**: {v}" for k, v in specs.items())
        feat_md = "\n".join(f"- {f}" for f in p["features"].split("\n") if f)
        emit(p["product_id"], "product", "public", p["title"], f"""
**Brand:** {p['brand']}  |  **Category:** {p['category']} / {p['subcategory']}  |  **SKU:** {p['sku']}
**Price:** {_money(p['price'])} (VAT inclusive)  |  **Rating:** {p['rating']} / 5 from {p['rating_count']} reviews

## Overview
{p['description']}

## Key features
{feat_md}

## Specifications
{spec_md}
""", product_id=p["product_id"], category=p["category"], brand=p["brand"], price=p["price"])

    for pid, (title, body) in POLICIES.items():
        emit(pid, "policy", "public", title, body)

    lines_by = defaultdict(list)
    for l in supply["quote_lines"]:
        lines_by[l["quote_id"]].append(l)
    for q in supply["quotations"]:
        party = cust[q["party_id"]]["name"] if q["kind"] == "customer_quote" else sup[q["party_id"]]["name"]
        rows = "\n".join(f"| {pmap[l['product_id']]['title'][:70]} | {l['qty']} | {_money(l['unit_price'])} | "
                         f"{_money(l['qty'] * l['unit_price'])} |" for l in lines_by[q["quote_id"]])
        who = "Customer" if q["kind"] == "customer_quote" else "Supplier"
        emit(q["quote_id"], "quotation", "internal", f"Quotation {q['quote_id']} - {party}", f"""
**Type:** {q['kind'].replace('_', ' ')}  |  **{who}:** {party} ({q['party_id']})
**Date:** {q['created_at']}  |  **Valid until:** {q['valid_until']}  |  **Status:** {q['status']}

| Item | Qty | Unit price | Line total |
|---|---|---|---|
{rows}

**Total:** {_money(q['total'])}

Notes: {q['notes']}
""", quote_id=q["quote_id"], party_id=q["party_id"], status=q["status"], total=q["total"])

    pol = defaultdict(list)
    for l in supply["po_lines"]:
        pol[l["po_id"]].append(l)
    for po in supply["purchase_orders"][-max_pos:]:
        s = sup[po["supplier_id"]]
        rows = "\n".join(f"| {pmap[l['product_id']]['sku']} | {pmap[l['product_id']]['title'][:70]} | {l['qty']} | "
                         f"{_money(l['unit_cost'])} |" for l in pol[po["po_id"]])
        emit(po["po_id"], "purchase_order", "internal", f"Purchase Order {po['po_id']} - {s['name']}", f"""
**Supplier:** {s['name']} ({s['supplier_id']}, {s['country']})  |  **Payment terms:** {s['payment_terms']}
**Created:** {po['created_at']}  |  **Expected delivery:** {po['expected_at']}  |  **Status:** {po['status']}

| SKU | Item | Qty | Unit cost |
|---|---|---|---|
{rows}

**PO total:** {_money(po['total_cost'])}
""", po_id=po["po_id"], supplier_id=po["supplier_id"], status=po["status"])

    items_by = defaultdict(list)
    for it in gen.items:
        items_by[it["order_id"]].append(it)
    orders = {o["order_id"]: o for o in gen.orders}
    for r in supply["receipts"][-max_receipts:]:
        o = orders[r["order_id"]]
        rows = "\n".join(f"| {pmap[i['product_id']]['title'][:70]} | {i['qty']} | {_money(i['line_total'])} |"
                         for i in items_by[o["order_id"]])
        emit(r["receipt_id"], "receipt", "owner", f"Receipt {r['receipt_id']} for order {o['order_id']}", f"""
**Order:** {o['order_id']}  |  **Customer ID:** {o['customer_id']}  |  **Date:** {r['issued_at']}
**Payment:** {r['payment_method']}  |  **Transaction ref:** {r['txn_ref']}  |  **Order status:** {o['status']}

| Item | Qty | Amount |
|---|---|---|
{rows}

Subtotal {_money(o['subtotal'])}, discount {_money(o['discount'])}, shipping {_money(o['shipping'])}.
**Total paid:** {_money(o['total'])} (includes VAT {_money(o['vat'])}). VAT no. 300000000000003.
""", order_id=o["order_id"], customer_id=o["customer_id"])

    with (out_dir / "manifest.jsonl").open("w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    return manifest


def build_rag_eval(gen: Generator, supply: dict, n_products: int = 30, seed: int = 3) -> list[dict]:
    """Questions with known gold documents + reference answers (for retrieval & faithfulness evals)."""
    rng = random.Random(seed)
    qs = [
        {"q": "How many days do I have to return a laptop?", "gold": ["policy-returns"], "answer": "7 days of delivery", "audience": "public"},
        {"q": "Is shipping free?", "gold": ["policy-shipping"], "answer": "Free on orders above 200 SAR, otherwise 25 SAR", "audience": "public"},
        {"q": "Can I pay in installments?", "gold": ["policy-payments"], "answer": "Tabby, split into 4 interest-free payments for 100-5,000 SAR", "audience": "public"},
        {"q": "What is the maximum order value for cash on delivery?", "gold": ["policy-payments"], "answer": "2,500 SAR", "audience": "public"},
        {"q": "How much does a 2 year TechCare+ plan cost for a 4500 SAR laptop?", "gold": ["policy-warranty"], "answer": "499 SAR", "audience": "public"},
        {"q": "What discount do companies get for 30 laptops?", "gold": ["policy-b2b"], "answer": "8%", "audience": "public"},
        {"q": "How long is a business quotation valid?", "gold": ["policy-b2b"], "answer": "14 days", "audience": "public"},
        {"q": "Can I upgrade the RAM in a MacBook Air?", "gold": ["faq-compatibility"], "answer": "No, memory is soldered", "audience": "public"},
        {"q": "Which GPU do I need for 1440p gaming?", "gold": ["faq-compatibility"], "answer": "RTX 4070", "audience": "public"},
        {"q": "Do you deliver same day in Riyadh?", "gold": ["policy-shipping"], "answer": "Yes, express same-day 49 SAR before 12 PM", "audience": "public"},
        {"q": "How long do you keep my chat history?", "gold": ["policy-privacy"], "answer": "12 months", "audience": "public"},
        {"q": "What happens to my data on a traded-in laptop?", "gold": ["policy-tradein"], "answer": "Wiped to NIST 800-88 with certificate", "audience": "public"},
        {"q": "If I find a lower price elsewhere will you match it?", "gold": ["policy-pricematch"], "answer": "Yes within 7 days at authorised Saudi retailers", "audience": "public"},
    ]
    for p in rng.sample(gen.products, n_products):
        specs = json.loads(p["specs_json"])
        t = p["title"].split(" - ")[0]
        qs.append({"q": f"How much is the {t}?" if rng.random() < 0.5 else f"What does the {t} cost?",
                   "gold": [p["product_id"]], "answer": f"{p['price']:,.0f} SAR", "audience": "public"})
        if "ram_gb" in specs and rng.random() < 0.5:
            qs.append({"q": f"How much RAM does the {p['title'][:60]} have?", "gold": [p["product_id"]],
                       "answer": f"{specs['ram_gb']}GB", "audience": "public"})
    for q in rng.sample([q for q in supply["quotations"] if q["kind"] == "customer_quote"], 8):
        qs.append({"q": f"What is the total and status of quotation {q['quote_id']}?", "gold": [q["quote_id"]],
                   "answer": f"{q['total']:,.2f} SAR, {q['status']}", "audience": "internal"})
    for po in rng.sample(supply["purchase_orders"][-150:], 6):
        qs.append({"q": f"When is purchase order {po['po_id']} expected to arrive?", "gold": [po["po_id"]],
                   "answer": po["expected_at"], "audience": "internal"})
    return qs
