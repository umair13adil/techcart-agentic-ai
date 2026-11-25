"""Product catalog - loaded from Hugging Face only.

Source: `McAuley-Lab/Amazon-Reviews-2023`, folder `raw_meta_Electronics/` (10 Parquet shards, ~2 GB).
We stream the shards (no full download), keep items whose Amazon category path is under
"Computers & Accessories", map them to the store's 15 categories, parse specs (RAM, SSD, screen, GPU ...)
from the `details` field and title, and cache the result to `.cache/catalog_hf.jsonl` so re-seeding is instant.

The dataset repo still carries a legacy loading script, which `datasets>=4` refuses to run, so we read the
Parquet files directly with the generic "parquet" builder over the `hf://` filesystem.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

USD_TO_SAR = 3.75
HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
HF_FILES = f"hf://datasets/{HF_REPO}/raw_meta_Electronics/full-*.parquet"

# --------------------------------------------------------------------------------------
# Taxonomy shared by every component (data gen, intent entities, recommender rules, UI)
# --------------------------------------------------------------------------------------
CATEGORIES = [
    "Laptops", "Desktops", "Monitors", "Keyboards", "Mice", "Headsets", "Webcams", "Docks & Hubs",
    "Storage", "Memory", "Graphics Cards", "Networking", "Bags & Sleeves", "Chargers & Power",
    "Cooling Pads",
]

# what typically gets bought *with / after* a category (drives basket synthesis + rec rules)
COMPLEMENTS: dict[str, list[str]] = {
    "Laptops": ["Mice", "Bags & Sleeves", "Docks & Hubs", "Headsets", "Storage", "Chargers & Power", "Monitors",
                "Cooling Pads"],
    "Desktops": ["Monitors", "Keyboards", "Mice", "Headsets", "Webcams", "Networking"],
    "Monitors": ["Docks & Hubs", "Keyboards", "Mice", "Webcams"],
    "Graphics Cards": ["Monitors", "Memory", "Storage", "Headsets"],
    "Keyboards": ["Mice", "Headsets"],
    "Mice": ["Keyboards", "Headsets"],
    "Headsets": ["Mice", "Webcams"],
    "Storage": ["Memory", "Docks & Hubs"],
    "Memory": ["Storage", "Graphics Cards"],
    "Docks & Hubs": ["Monitors", "Chargers & Power"],
    "Webcams": ["Headsets", "Monitors"],
    "Networking": ["Storage"],
    "Bags & Sleeves": ["Chargers & Power", "Mice"],
    "Chargers & Power": ["Bags & Sleeves"],
    "Cooling Pads": ["Mice", "Headsets"],
}

# "durable" categories you rarely re-buy right after purchase (recommender suppresses them)
DURABLE = {"Laptops", "Desktops", "Monitors", "Graphics Cards"}

# How many products to keep per category (laptops get more: they anchor most baskets)
DEFAULT_QUOTA = {c: 28 for c in CATEGORIES} | {"Laptops": 70, "Monitors": 35, "Desktops": 25}

# --------------------------------------------------------------------------------------
# Amazon category path -> store category. Matched leaf-first, so
# ["Laptop Accessories", "Bags, Cases & Sleeves", "Sleeves"] lands on "Bags & Sleeves", not "Laptops".
# --------------------------------------------------------------------------------------
_SEGMENT_RULES = [
    ("Bags & Sleeves", r"bags|backpacks|sleeves|cases|messenger"),
    ("Cooling Pads", r"cooling pads|cooler|cooling"),
    ("Chargers & Power", r"chargers|adapters|power banks|uninterruptible|ups|batteries|power supplies"),
    ("Docks & Hubs", r"docking|docks|usb hubs|hubs"),
    ("Keyboards", r"keyboards"),
    ("Mice", r"^mice$|mice|trackballs"),
    ("Headsets", r"headsets|headphones"),
    ("Webcams", r"webcams"),
    ("Monitors", r"^monitors$|monitors"),
    ("Graphics Cards", r"graphics cards"),
    ("Memory", r"^memory$|^ram$"),
    ("Storage", r"solid state drives|hard drives|usb flash|flash drives|external drives|data storage"),
    ("Networking", r"routers|mesh|network adapters|wireless access points|networking"),
    ("Desktops", r"^desktops$|^towers$|all-in-ones?|^minis$|mini pcs?"),
    ("Laptops", r"^(traditional )?laptops$|^2 in 1 laptops$|^chromebooks?$|^laptop computers$"),
]
_TITLE_EXCLUDE = re.compile(r"\b(renewed|refurbished|used|replacement|skin|sticker|decal|screen protector|"
                            r"keyboard cover|battery for|for parts)\b", re.I)


def map_category(categories: list[str], title: str) -> str | None:
    for seg in reversed([c.strip().lower() for c in categories or []]):
        for cat, pat in _SEGMENT_RULES:
            if re.search(pat, seg):
                return cat
    return None


def infer_subcategory(cat: str, title: str, specs: dict) -> str:
    """Store sub-categories (used by personas, search boosts and B2B quotes) inferred from title + specs."""
    t = title.lower()
    has = lambda *w: any(x in t for x in w)  # noqa: E731
    if cat == "Laptops":
        if has("gaming", "rog", "legion", "omen", "nitro", "predator", "alienware", "tuf", "katana", "victus"):
            return "gaming"
        if has("thinkpad", "latitude", "elitebook", "probook", "expertbook", "travelmate", "business", "vpro"):
            return "business"
        if has("2-in-1", "2 in 1", "x360", "flip", "convertible", "yoga", "detachable"):
            return "2-in-1"
        if has("macbook pro", "xps 15", "xps 16", "proart", "creator", "studio", "zbook", "precision"):
            return "creator"
        if re.search(r"rtx\s?\d{4}", t):
            return "gaming"
        if has("macbook air", "zenbook", "swift", "spectre", "gram", "thin", "slim", "ultra", "surface laptop"):
            return "ultrabook"
        return "student"
    if cat == "Monitors":
        if (specs.get("refresh_hz") or 0) >= 144 or has("gaming", "144hz", "165hz", "240hz"):
            return "gaming"
        if has("4k", "uhd", "3840", "usb-c", "usb c", "thunderbolt", "color accurate", "designer"):
            return "creator"
        return "office"
    if cat in ("Mice", "Headsets"):
        return "gaming" if has("gaming", "esports", "rgb") else "office"
    if cat == "Keyboards":
        return "mechanical" if has("mechanical", "hot-swap", "gaming") else "office"
    if cat == "Webcams":
        return "streaming" if has("4k", "stream", "60fps") else "office"
    if cat == "Docks & Hubs":
        return "dock" if has("dock", "thunderbolt") else "hub"
    if cat == "Storage":
        if has("nvme", "m.2", "internal", "sata iii", "2.5 inch", "2.5\"") and not has("portable", "external"):
            return "internal ssd"
        if has("ssd", "solid state") and has("portable", "external"):
            return "external ssd"
        if has("hard drive", "hdd", "external"):
            return "external hdd"
        return "flash"
    if cat == "Memory":
        return "laptop ram" if has("sodimm", "so-dimm", "laptop", "notebook") else "desktop ram"
    if cat == "Graphics Cards":
        return "amd" if has("radeon", "amd") else "nvidia"
    if cat == "Networking":
        return "mesh" if has("mesh") else "router" if has("router") else "adapter"
    if cat == "Bags & Sleeves":
        return "backpack" if has("backpack") else "sleeve" if has("sleeve") else "bag"
    if cat == "Chargers & Power":
        return "power bank" if has("power bank", "portable charger") else "ups" if has("ups", "battery backup") \
            else "charger"
    if cat == "Desktops":
        if has("gaming", "rtx", "radeon rx"):
            return "gaming desktop"
        if has("all-in-one", "all in one", "imac"):
            return "all-in-one"
        if has("mini", "nuc", "mac mini"):
            return "mini pc"
        return "business desktop"
    return cat.lower()


# --------------------------------------------------------------------------------------
# Spec parsing from Amazon `details` + title
# --------------------------------------------------------------------------------------
def _gb(text: str) -> int | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", str(text).lower())
    if not m:
        return None
    v = float(m.group(1))
    return int(v * 1000) if m.group(2) == "tb" else int(v)


def parse_specs(details: dict, title: str) -> dict:
    d = {k.lower(): str(v) for k, v in (details or {}).items()}
    t = title.lower()
    s: dict = {}
    ram = d.get("ram") or d.get("ram memory installed size") or d.get("computer memory size")
    if ram and _gb(ram):
        s["ram_gb"] = _gb(ram)
    elif m := re.search(r"(\d{1,3})\s?gb\s*(?:ddr\d\w*\s*)?(?:ram|memory)", t):
        s["ram_gb"] = int(m.group(1))
    disk = d.get("hard disk size") or d.get("hard drive") or d.get("digital storage capacity") \
        or d.get("memory storage capacity") or d.get("flash memory size")
    if disk and _gb(disk):
        s["storage_gb"] = _gb(disk)
    elif m := re.search(r"(\d+(?:\.\d+)?)\s?(tb|gb)\s*(?:pcie\s*)?(?:nvme\s*)?(?:ssd|emmc|hdd|storage)", t):
        s["storage_gb"] = _gb(m.group(0))
    scr = d.get("screen size") or d.get("standing screen display size") or d.get("display size")
    m = re.search(r"(\d{2}(?:\.\d)?)", scr or "") or re.search(r"(\d{2}(?:\.\d)?)\s?(?:\"|”|''|-inch|\s?inch|\s?in\b)", t)
    if m and 10 <= float(m.group(1)) <= 49:
        s["screen_in"] = float(m.group(1))
    if m := re.search(r"(\d{2,3})\s?hz", (d.get("refresh rate", "") + " " + t)):
        s["refresh_hz"] = int(m.group(1))
    gpu = d.get("graphics coprocessor") or d.get("graphics card description")
    m = re.search(r"(rtx\s?\d{4}(?:\s?ti|\s?super)?|gtx\s?\d{4}|rx\s?\d{4}(?:\s?xt)?|arc\s?a\d{3})", (gpu or "") + " " + t)
    if m:
        s["gpu"] = m.group(1).upper().replace("  ", " ")
    elif gpu:
        s["gpu"] = gpu[:60]
    cpu = d.get("cpu model") or d.get("processor") or d.get("cpu manufacturer")
    m = re.search(r"(core\s?(?:ultra\s?)?i?[3579][\w\-]*|ryzen\s?\d\s?\w*|apple m\d\w*|m[1-4]\s?(?:pro|max)?\s?chip|"
                  r"celeron\s?\w*|snapdragon[\w\s]*)", t)
    if cpu:
        s["cpu"] = cpu[:60]
    elif m:
        s["cpu"] = m.group(1).strip()[:60]
    if "fits" in t or "up to" in t:
        if m := re.search(r"(?:up to|fits)\s*(\d{2}(?:\.\d)?)", t):
            s["fits_in"] = float(m.group(1))
    conn = d.get("connectivity technology", "") + " " + t
    if re.search(r"wireless|bluetooth|2\.4\s?g", conn, re.I):
        s["wireless"] = True
    for k in ("brand", "color", "model name", "operating system", "item weight", "connectivity technology"):
        if d.get(k):
            s[k.replace(" ", "_")] = d[k][:60]
    return s


def _parse_price(p) -> float | None:
    if p is None:
        return None
    try:
        return float(str(p).replace("$", "").replace(",", "").split()[0])
    except (ValueError, IndexError):
        return None


def _first_image(images) -> str | None:
    """`images` is a dict of lists in the Parquet export (hi_res/large/thumb) or a list of dicts in JSONL."""
    if isinstance(images, dict):
        for key in ("large", "hi_res", "thumb"):
            for u in images.get(key) or []:
                if u:
                    return u
    elif isinstance(images, list):
        for im in images:
            if isinstance(im, dict):
                u = im.get("large") or im.get("hi_res") or im.get("thumb")
                if u:
                    return u
    return None


def from_amazon(rec: dict, rng: random.Random) -> dict | None:
    """Map one Amazon-Reviews-2023 metadata record to a TechCart product (or None if unsuitable)."""
    cats = list(rec.get("categories") or [])
    title = (rec.get("title") or "").strip()
    if "Computers & Accessories" not in cats or not title or _TITLE_EXCLUDE.search(title):
        return None
    cat, usd = map_category(cats, title), _parse_price(rec.get("price"))
    if not cat or not usd or usd < 5 or usd > 6000:
        return None
    if cat in ("Laptops", "Desktops") and usd < 150:  # filters out accessories mis-filed as computers
        return None
    details = rec.get("details") or {}
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = {}
    store = (rec.get("store") or "").replace("Visit the ", "").replace(" Store", "").strip()
    brand = (details.get("Brand") or store or title.split()[0]).strip()[:40]
    specs = parse_specs(details, title)
    if cat in ("Storage", "Memory"):
        gb = specs.pop("storage_gb", None) or specs.get("ram_gb") or _gb(title)
        if gb:
            specs["capacity_gb"] = gb
    features = [f for f in (rec.get("features") or []) if f][:8]
    desc = " ".join(d for d in (rec.get("description") or []) if d)[:1500] or (features[0] if features else title)
    price = float(round(usd * USD_TO_SAR / 5) * 5 - 1)  # retail-style 1,499 pricing in SAR (VAT incl.)
    return {
        "product_id": f"AMZ{rec['parent_asin']}", "sku": rec["parent_asin"], "title": title[:180], "brand": brand,
        "category": cat, "subcategory": infer_subcategory(cat, title, specs), "price": max(price, 9.0),
        "cost": round(max(price, 9.0) * rng.uniform(0.72, 0.86), 2),
        "rating": float(rec.get("average_rating") or 0) or round(rng.uniform(3.8, 4.7), 1),
        "rating_count": int(rec.get("rating_number") or 0), "specs_json": json.dumps(specs),
        "features": "\n".join(features), "description": desc, "source": "hf:amazon-reviews-2023",
        "image_url": _first_image(rec.get("images")),
    }


def _iter_hf() -> Iterable[dict]:
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files={"full": HF_FILES}, split="full", streaming=True)
    yield from ds


def _cache_path() -> Path:
    from ..config import ROOT

    return Path(os.environ.get("TECHCART_CATALOG_CACHE", ROOT / ".cache" / "catalog_hf.jsonl"))


def load_catalog(quota: dict[str, int] | None = None, scan_limit: int = 1_500_000, seed: int = 7,
                 refresh: bool = False, records: Iterable[dict] | None = None) -> list[dict]:
    """Stream Amazon Electronics metadata from Hugging Face until every category quota is filled (or
    `scan_limit` records were read), then cache. `records` lets tests inject an iterator in the same schema."""
    cache = _cache_path()
    if cache.exists() and not refresh and records is None:
        out = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
        log.info("Loaded %d products from cache %s (use --refresh to re-download)", len(out), cache)
        return out
    quota = quota or DEFAULT_QUOTA
    rng = random.Random(seed)
    counts = {c: 0 for c in CATEGORIES}
    seen_titles: set[str] = set()
    out: list[dict] = []
    it = records if records is not None else _iter_hf()
    for i, rec in enumerate(it):
        if i >= scan_limit or all(counts[c] >= quota.get(c, 0) for c in CATEGORIES):
            break
        if i and i % 100_000 == 0:
            log.info("scanned %d records, kept %d: %s", i, len(out), counts)
        p = from_amazon(rec, rng)
        if not p or counts[p["category"]] >= quota.get(p["category"], 0):
            continue
        key = re.sub(r"[^a-z0-9]", "", p["title"].lower())[:50]  # drop colour/size duplicates
        if key in seen_titles:
            continue
        seen_titles.add(key)
        counts[p["category"]] += 1
        out.append(p)
    if len(out) < 50:
        raise RuntimeError(f"Only {len(out)} usable products found on Hugging Face ({HF_REPO}). "
                           "Check your internet connection / HF access and retry.")
    missing = [c for c in CATEGORIES if counts[c] == 0]
    if missing:
        log.warning("No products found for %s (raise --scan-limit to search further)", missing)
    if records is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("\n".join(json.dumps(p) for p in out) + "\n")
    log.info("Catalog: %d products %s", len(out), counts)
    return out
