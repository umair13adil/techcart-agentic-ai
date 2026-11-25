"""Query understanding: intent classification + entity extraction, as a cost-aware cascade.

    tier 1  TF-IDF (word + char n-grams) + calibrated Logistic Regression      ~1 ms, free
    tier 2  Azure OpenAI structured output (only if tier-1 confidence < tau)   ~600 ms, $$

Entities (category, brand, budget, use case, specs, order id) come from a deterministic extractor that the
LLM tier can override. Every decision is logged (search_log) -> drift monitoring + relabelling queue.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from ..data.catalog import CATEGORIES
from .intent_data import TEST_SET, generate

CONFIDENCE_THRESHOLD = 0.55

CATEGORY_LEXICON = {
    "Laptops": ["laptop", "lappy", "notebook", "macbook", "ultrabook", "thinkpad", "chromebook", "zenbook",
                "xps", "latitude", "elitebook", "legion", "zephyrus", "2-in-1", "surface laptop", "لابتوب"],
    "Desktops": ["desktop", "mini pc", "all-in-one", "imac", "mac mini", "tower", "pc build"],
    "Monitors": ["monitor", "screen", "display", "curved"],
    "Keyboards": ["keyboard", "keychron", "mx keys"],
    "Mice": ["mouse", "mice", "g502", "mx master", "deathadder"],
    "Headsets": ["headset", "headphone", "earphone", "arctis"],
    "Webcams": ["webcam", "web cam", "camera for calls", "facecam", "brio"],
    "Docks & Hubs": ["dock", "docking", "hub", "usb-c hub", "thunderbolt"],
    "Storage": ["ssd", "hard drive", "hdd", "storage", "nvme", "portable drive"],
    "Memory": ["ram", "memory", "ddr5", "sodimm"],
    "Graphics Cards": ["graphics card", "gpu", "video card", "radeon", "rtx 40"],
    "Networking": ["router", "wifi", "wi-fi", "mesh", "ethernet"],
    "Bags & Sleeves": ["bag", "backpack", "sleeve", "case"],
    "Chargers & Power": ["charger", "power bank", "adapter", "ups", "powerbank"],
    "Cooling Pads": ["cooling pad", "cooler pad", "laptop cooler"],
}
USE_CASES = {
    "gaming": ["gaming", "game", "gamer", "fps", "esports", "ps5", "qaming", "قيمنق"],
    "programming": ["programming", "coding", "developer", "software", "cs degree", "computer science", "code"],
    "creative": ["video editing", "premiere", "editing", "design", "photoshop", "3d", "creator", "youtube",
                 "streaming", "content"],
    "office": ["office", "excel", "work", "business", "zoom", "teams", "calls", "meetings"],
    "student": ["student", "university", "uni", "college", "school", "class", "kids"],
    "travel": ["travel", "portable", "light", "lightweight", "thin", "battery life"],
    "ai_ml": ["machine learning", "deep learning", "ai models", "llm", "cuda"],
}
BRANDS = ["dell", "hp", "lenovo", "asus", "acer", "apple", "msi", "microsoft", "logitech", "razer", "samsung", "lg",
          "benq", "keychron", "steelseries", "hyperx", "jabra", "sony", "anker", "ugreen", "caldigit", "wd",
          "crucial", "sandisk", "seagate", "kingston", "corsair", "g.skill", "gigabyte", "sapphire", "tp-link",
          "targus", "samsonite", "tomtoc", "incase", "apc", "cooler master", "klim", "elgato", "intel"]
BRAND_ALIASES = {"macbook": "apple", "mac": "apple", "thinkpad": "lenovo", "legion": "lenovo", "xps": "dell",
                 "alienware": "dell", "rog": "asus", "zenbook": "asus", "omen": "hp", "surface": "microsoft"}


class Entities(BaseModel):
    category: str | None = Field(None, description=f"one of {CATEGORIES}")
    brand: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    use_case: str | None = Field(None, description=f"one of {list(USE_CASES)}")
    ram_gb: int | None = None
    storage_gb: int | None = None
    screen_in: float | None = None
    gpu: str | None = None
    quantity: int | None = None
    order_id: str | None = None
    keywords: list[str] = Field(default_factory=list)


class QueryParse(BaseModel):
    """Structured output schema for the LLM tier."""
    intent: Literal["product_search", "comparison", "recommendation", "order_status", "return_refund",
                    "policy_question", "purchase", "quote_request", "other"]
    entities: Entities
    rewritten_query: str = Field(description="clean English search query for retrieval")


@dataclass
class Understanding:
    query: str
    intent: str
    confidence: float
    entities: dict
    rewritten_query: str
    tier: str  # "classifier" | "llm"
    alternatives: list[tuple[str, float]] = field(default_factory=list)

    def dict(self) -> dict:
        return asdict(self)


def _num(s: str) -> float:
    s = s.lower().replace(",", "").strip()
    mult = 1000 if s.endswith("k") else 1
    return float(s.rstrip("k")) * mult


def extract_entities(q: str) -> Entities:
    t = " " + q.lower() + " "
    e = Entities()
    # head-noun heuristic: the product is named before "for/with/under..." and is the right-most noun there
    # ("laptop bag" -> bag, "4k monitor for my macbook" -> monitor, brand apple is NOT a constraint)
    head = re.split(r"\b(?:for|with|under|below|to|that|compatible|less|between|around|in)\b", t, maxsplit=1)[0]

    def find_cat(seg: str) -> str | None:
        best, pos = None, -1
        for cat, words in CATEGORY_LEXICON.items():
            for w in words:
                i = seg.rfind(w)
                if i > pos:
                    best, pos = cat, i
        return best

    head_cat = find_cat(head)
    e.category = head_cat or find_cat(t)

    def find_brand(seg: str) -> str | None:
        for b in sorted(BRANDS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(b)}\b", seg):
                return b
        for a, b in BRAND_ALIASES.items():
            if re.search(rf"\b{a}s?\b", seg):
                return b
        return None

    # brand only from the head phrase, unless the product itself is named after the preposition
    e.brand = find_brand(head) if head_cat else find_brand(t)
    for uc, words in USE_CASES.items():
        if any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", t) for w in words):
            e.use_case = uc
            break
    num = r"(\d[\d,]*\.?\d*\s?k?)"
    if m := re.search(rf"between\s+{num}\s+(?:and|-|to)\s+{num}", t):
        e.min_price, e.max_price = _num(m.group(1)), _num(m.group(2))
    elif m := re.search(rf"(?:under|below|less than|max|maximum|up to|upto|<|nothing over|not more than|budget(?: of)?)\s*{num}", t):
        e.max_price = _num(m.group(1))
    elif m := re.search(rf"(?:over|above|more than|at least|min)\s*{num}\s*(?:sar|riyal|sr)", t):
        e.min_price = _num(m.group(1))
    elif m := re.search(rf"(?:around|about|~)\s*{num}", t):
        v = _num(m.group(1))
        e.min_price, e.max_price = v * 0.85, v * 1.15
    elif m := re.search(rf"{num}\s*(?:sar|riyal|sr)\b", t):
        e.max_price = _num(m.group(1))
    if m := re.search(r"(\d{1,3})\s?gb\s*(?:of\s*)?(?:ram|memory|ddr)", t):
        e.ram_gb = int(m.group(1))
    if m := re.search(r"(\d{1,2})\s?tb\b", t):
        e.storage_gb = int(m.group(1)) * 1000
    elif m := re.search(r"(\d{3,4})\s?gb\s*(?:ssd|storage|nvme)", t):
        e.storage_gb = int(m.group(1))
    if m := re.search(r"(\d{2}(?:\.\d)?)\s?(?:\"|inch|in\b|”)", t):
        e.screen_in = float(m.group(1))
    if m := re.search(r"\b(rtx\s?\d{4}|rx\s?\d{4})\b", t):
        e.gpu = m.group(1).upper().replace(" ", " ")
    if m := re.search(r"\b(o\d{7})\b", t):
        e.order_id = m.group(1).upper()
    if m := re.search(r"\b(\d{1,4})\s+(?:[a-z\-]+\s+){0,3}?(?:units|laptops|monitors|desktops|pcs|thinkpads|latitudes|"
                       r"machines|devices|docks|notebooks|headsets|of the)\b", t):
        e.quantity = int(m.group(1))
    e.keywords = [w for w in re.findall(r"[a-z0-9\-\+]{3,}", t) if not w.isdigit() and w not in {"the", "and", "for", "with", "need",
                                                                              "want", "show", "looking", "under"}][:8]
    return e


def build_pipeline() -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    clf = CalibratedClassifierCV(LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"), cv=3)
    return Pipeline([("features", feats), ("clf", clf)])


def train_and_evaluate(seed: int = 11) -> tuple[Pipeline, dict, str]:
    data = generate(seed=seed)
    X, y = [d[0] for d in data], [d[1] for d in data]
    pipe = build_pipeline().fit(X, y)
    Xt, yt = [t[0] for t in TEST_SET], [t[1] for t in TEST_SET]
    proba = pipe.predict_proba(Xt)
    pred = pipe.classes_[proba.argmax(1)]
    conf = proba.max(1)
    confident = conf >= CONFIDENCE_THRESHOLD
    metrics = {
        "test_accuracy": round(accuracy_score(yt, pred), 4),
        "test_macro_f1": round(f1_score(yt, pred, average="macro"), 4),
        # cascade view: what the classifier handles alone vs. what would be routed to the LLM
        "coverage_at_tau": round(float(confident.mean()), 4),
        "accuracy_when_confident": round(float((np.array(yt)[confident] == pred[confident]).mean()), 4)
        if confident.any() else 0.0,
        "train_size": len(X), "test_size": len(Xt), "tau": CONFIDENCE_THRESHOLD,
    }
    report = classification_report(yt, pred, zero_division=0)
    return pipe, metrics, report


class QueryUnderstander:
    def __init__(self, pipeline: Pipeline, threshold: float = CONFIDENCE_THRESHOLD, use_llm: bool = True):
        self.pipe, self.threshold, self.use_llm = pipeline, threshold, use_llm

    def classify(self, q: str) -> tuple[str, float, list[tuple[str, float]]]:
        p = self.pipe.predict_proba([q])[0]
        order = np.argsort(-p)
        cls = self.pipe.classes_
        return cls[order[0]], float(p[order[0]]), [(cls[i], round(float(p[i]), 3)) for i in order[1:3]]

    def understand(self, q: str) -> Understanding:
        intent, conf, alts = self.classify(q)
        ents = extract_entities(q)
        if ents.order_id and intent not in ("order_status", "return_refund"):
            intent, conf = "order_status", max(conf, 0.6)
        u = Understanding(q, intent, round(conf, 3), ents.model_dump(), q, "classifier", alts)
        if conf < self.threshold and self.use_llm:
            try:
                u = self._llm(q, ents) or u
            except Exception:  # LLM down / no key -> graceful degradation to tier-1 answer
                pass
        return u

    def _llm(self, q: str, ents: Entities) -> Understanding | None:
        from ..llm import chat_model

        llm = chat_model().with_structured_output(QueryParse)
        out: QueryParse = llm.invoke([
            ("system", "You parse shopper queries for a Saudi computer & accessories store (prices in SAR). "
                       "Return the intent, entities and a clean English search query. Translate Arabic. "
                       f"Pre-extracted hints (may be wrong): {ents.model_dump_json(exclude_none=True)}"),
            ("user", q),
        ])
        merged = {**ents.model_dump(), **{k: v for k, v in out.entities.model_dump().items() if v not in (None, [])}}
        return Understanding(q, out.intent, 0.9, merged, out.rewritten_query, "llm")
