"""Product catalog sources.

Order of preference (see `load_catalog`):
  1. Hugging Face  - McAuley-Lab/Amazon-Reviews-2023, config `raw_meta_Electronics`
                     (streamed, filtered to "Computers & Accessories"). Real titles, features,
                     descriptions, prices, ratings.
  2. UCSD mirror   - same data as a gzipped JSONL, streamed over HTTP (no `datasets` needed).
  3. Offline       - a deterministic, spec-realistic catalog generated below so the project runs
                     anywhere (CI, air-gapped laptops, interview demo with bad Wi-Fi).
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import random
import re
from typing import Iterable

log = logging.getLogger(__name__)

USD_TO_SAR = 3.75

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


# --------------------------------------------------------------------------------------
# Offline catalog
# --------------------------------------------------------------------------------------
LAPTOP_LINES = [
    # brand, model, subcategory, base_usd
    ("Dell", "XPS 13", "ultrabook", 1099), ("Dell", "XPS 15", "creator", 1699),
    ("Dell", "Latitude 5450", "business", 1249), ("Dell", "Inspiron 15 3530", "student", 579),
    ("Dell", "Alienware m16 R2", "gaming", 1899), ("HP", "Spectre x360 14", "2-in-1", 1449),
    ("HP", "EliteBook 840 G11", "business", 1399), ("HP", "Pavilion 15", "student", 649),
    ("HP", "OMEN 16", "gaming", 1399), ("HP", "Victus 15", "gaming", 849),
    ("Lenovo", "ThinkPad X1 Carbon Gen 12", "business", 1799), ("Lenovo", "ThinkPad T14 Gen 5", "business", 1199),
    ("Lenovo", "IdeaPad Slim 5", "student", 699), ("Lenovo", "Legion Pro 5", "gaming", 1599),
    ("Lenovo", "Yoga 7i 2-in-1", "2-in-1", 899), ("ASUS", "ROG Zephyrus G14", "gaming", 1599),
    ("ASUS", "TUF Gaming A15", "gaming", 999), ("ASUS", "Zenbook 14 OLED", "ultrabook", 949),
    ("ASUS", "Vivobook 15", "student", 549), ("ASUS", "ProArt P16", "creator", 2199),
    ("Acer", "Swift Go 14", "ultrabook", 799), ("Acer", "Nitro V 15", "gaming", 799),
    ("Acer", "Aspire 5", "student", 529), ("Apple", "MacBook Air 13 (M3)", "ultrabook", 1099),
    ("Apple", "MacBook Air 15 (M3)", "ultrabook", 1299), ("Apple", "MacBook Pro 14 (M4)", "creator", 1599),
    ("Apple", "MacBook Pro 16 (M4 Pro)", "creator", 2499), ("MSI", "Katana 15", "gaming", 999),
    ("MSI", "Prestige 14 AI", "ultrabook", 1099), ("Microsoft", "Surface Laptop 7", "ultrabook", 999),
    ("Microsoft", "Surface Pro 11", "2-in-1", 999),
]
CPU_BY_SUB = {
    "gaming": ["Intel Core i7-14650HX", "AMD Ryzen 7 8845HS", "Intel Core i9-14900HX", "AMD Ryzen 9 8945HS"],
    "ultrabook": ["Intel Core Ultra 7 155H", "Intel Core Ultra 5 125U", "AMD Ryzen 7 8840U", "Snapdragon X Plus"],
    "business": ["Intel Core Ultra 5 135U", "Intel Core Ultra 7 165U", "AMD Ryzen 7 PRO 8840U"],
    "student": ["Intel Core i5-1335U", "AMD Ryzen 5 7530U", "Intel Core i3-1315U"],
    "2-in-1": ["Intel Core Ultra 7 155U", "Snapdragon X Elite", "Intel Core Ultra 5 125U"],
    "creator": ["Intel Core Ultra 9 185H", "Intel Core i9-13900H", "AMD Ryzen AI 9 HX 370"],
}
GPU_BY_SUB = {
    "gaming": ["NVIDIA GeForce RTX 4060 8GB", "NVIDIA GeForce RTX 4070 8GB", "NVIDIA GeForce RTX 4080 12GB",
               "NVIDIA GeForce RTX 4050 6GB"],
    "creator": ["NVIDIA GeForce RTX 4060 8GB", "NVIDIA GeForce RTX 4070 8GB"],
}
LAPTOP_CONFIGS = [(8, 256, 0), (16, 512, 180), (16, 1024, 320), (32, 1024, 620)]
USE_CASE_COPY = {
    "gaming": "Built for AAA gaming and streaming with a high-refresh display and advanced cooling.",
    "ultrabook": "Thin, light and all-day battery life - ideal for travel, office work and studying.",
    "business": "Enterprise-grade security (TPM 2.0, fingerprint, IR camera), durability and manageability for teams.",
    "student": "Great value for classes, assignments, browsing and light programming.",
    "2-in-1": "Convertible touchscreen with pen support for note-taking, design sketches and presentations.",
    "creator": "Colour-accurate display and powerful CPU/GPU for video editing, 3D and software development.",
}

ACCESSORY_LINES = [
    # category, subcategory, brand, model, base_usd, spec dict, blurb
    ("Monitors", "gaming", "ASUS", "TUF Gaming VG27AQ 27\" QHD 165Hz", 299, {"size_in": 27, "resolution": "2560x1440", "refresh_hz": 165, "panel": "IPS"}, "Fast IPS gaming monitor with G-Sync compatibility."),
    ("Monitors", "gaming", "Samsung", "Odyssey G5 32\" QHD 144Hz Curved", 329, {"size_in": 32, "resolution": "2560x1440", "refresh_hz": 144, "panel": "VA"}, "1000R curved VA panel for immersive gaming."),
    ("Monitors", "gaming", "LG", "UltraGear 27GR95QE 27\" OLED 240Hz", 899, {"size_in": 27, "resolution": "2560x1440", "refresh_hz": 240, "panel": "OLED"}, "OLED with 0.03ms response for competitive esports."),
    ("Monitors", "office", "Dell", "P2425H 24\" FHD", 199, {"size_in": 24, "resolution": "1920x1080", "refresh_hz": 100, "panel": "IPS"}, "Comfortview Plus low blue light office monitor."),
    ("Monitors", "office", "HP", "E24 G5 24\" FHD", 189, {"size_in": 24, "resolution": "1920x1080", "refresh_hz": 75, "panel": "IPS"}, "Ergonomic stand with height, tilt and pivot."),
    ("Monitors", "creator", "Dell", "UltraSharp U2723QE 27\" 4K USB-C Hub", 579, {"size_in": 27, "resolution": "3840x2160", "refresh_hz": 60, "panel": "IPS Black", "usb_c_watts": 90}, "4K IPS Black with 90W USB-C single-cable docking."),
    ("Monitors", "creator", "LG", "32UN880 Ergo 32\" 4K", 649, {"size_in": 32, "resolution": "3840x2160", "refresh_hz": 60, "panel": "IPS", "usb_c_watts": 60}, "Clamp-mounted ergo arm, DCI-P3 95%."),
    ("Monitors", "creator", "BenQ", "PD2706U 27\" 4K Designer", 549, {"size_in": 27, "resolution": "3840x2160", "refresh_hz": 60, "panel": "IPS"}, "Factory calibrated, Mac-friendly colour modes."),
    ("Keyboards", "mechanical", "Logitech", "G Pro X TKL Lightspeed", 199, {"switches": "GX Brown tactile", "wireless": True}, "Tenkeyless wireless esports keyboard."),
    ("Keyboards", "mechanical", "Keychron", "K8 Pro Wireless Mechanical", 109, {"switches": "Gateron Red", "wireless": True, "layout": "TKL"}, "Mac/Windows hot-swappable mechanical keyboard."),
    ("Keyboards", "mechanical", "Razer", "BlackWidow V4 Pro", 229, {"switches": "Razer Green", "wireless": False}, "RGB gaming keyboard with command dial."),
    ("Keyboards", "office", "Logitech", "MX Keys S", 109, {"wireless": True, "backlit": True}, "Low-profile backlit keys, multi-device Easy-Switch."),
    ("Keyboards", "office", "Microsoft", "Ergonomic Keyboard", 59, {"wireless": False}, "Split ergonomic layout with cushioned palm rest."),
    ("Mice", "gaming", "Logitech", "G502 X Plus", 159, {"dpi": 25600, "wireless": True, "weight_g": 106}, "Hybrid optical-mechanical switches, LIGHTSPEED wireless."),
    ("Mice", "gaming", "Razer", "DeathAdder V3 Pro", 149, {"dpi": 30000, "wireless": True, "weight_g": 63}, "Ultra-light ergonomic esports mouse."),
    ("Mice", "gaming", "SteelSeries", "Rival 3", 29, {"dpi": 8500, "wireless": False}, "Budget wired gaming mouse."),
    ("Mice", "office", "Logitech", "MX Master 3S", 99, {"dpi": 8000, "wireless": True, "quiet_clicks": True}, "Quiet clicks, MagSpeed scroll, works on glass."),
    ("Mice", "office", "Logitech", "M240 Silent Bluetooth", 29, {"wireless": True}, "Compact silent mouse for laptops."),
    ("Mice", "office", "Apple", "Magic Mouse", 79, {"wireless": True}, "Multi-touch surface for macOS gestures."),
    ("Headsets", "gaming", "HyperX", "Cloud III Wireless", 169, {"wireless": True, "battery_h": 120}, "120-hour battery, DTS spatial audio."),
    ("Headsets", "gaming", "SteelSeries", "Arctis Nova 7", 179, {"wireless": True, "battery_h": 38}, "Simultaneous 2.4GHz + Bluetooth."),
    ("Headsets", "office", "Jabra", "Evolve2 55", 299, {"wireless": True, "anc": True, "teams_certified": True}, "Microsoft Teams certified ANC headset."),
    ("Headsets", "office", "Sony", "WH-1000XM5", 399, {"wireless": True, "anc": True}, "Industry-leading noise cancelling headphones."),
    ("Headsets", "office", "Logitech", "H390 USB", 39, {"wireless": False}, "Plug-and-play USB headset for calls and classes."),
    ("Webcams", "streaming", "Logitech", "Brio 4K", 199, {"resolution": "4K", "hdr": True}, "4K HDR webcam with Windows Hello."),
    ("Webcams", "office", "Logitech", "C920s HD Pro", 69, {"resolution": "1080p"}, "Reliable 1080p webcam with privacy shutter."),
    ("Webcams", "streaming", "Elgato", "Facecam MK.2", 149, {"resolution": "1080p60"}, "Studio-grade webcam for streamers."),
    ("Docks & Hubs", "dock", "CalDigit", "TS4 Thunderbolt 4 Dock", 399, {"ports": 18, "pd_watts": 98}, "18 ports, 98W charging, dual 4K displays."),
    ("Docks & Hubs", "dock", "Dell", "WD22TB4 Thunderbolt Dock", 289, {"pd_watts": 130}, "Business dock with remote management."),
    ("Docks & Hubs", "hub", "Anker", "PowerExpand 8-in-1 USB-C Hub", 79, {"ports": 8, "pd_watts": 100}, "4K HDMI, SD card, Ethernet, 100W pass-through."),
    ("Docks & Hubs", "hub", "UGREEN", "Revodok 7-in-1 USB-C Hub", 39, {"ports": 7}, "Compact travel hub."),
    ("Storage", "internal ssd", "Samsung", "990 PRO 2TB NVMe SSD", 179, {"capacity_gb": 2000, "interface": "PCIe 4.0 NVMe"}, "Up to 7,450 MB/s reads for gaming and creators."),
    ("Storage", "internal ssd", "WD", "Black SN850X 1TB NVMe SSD", 99, {"capacity_gb": 1000, "interface": "PCIe 4.0 NVMe"}, "Game Mode 2.0 for faster load times."),
    ("Storage", "internal ssd", "Crucial", "P3 Plus 1TB NVMe SSD", 69, {"capacity_gb": 1000, "interface": "PCIe 4.0 NVMe"}, "Affordable PCIe 4.0 upgrade."),
    ("Storage", "external ssd", "Samsung", "T7 Shield 2TB Portable SSD", 169, {"capacity_gb": 2000, "interface": "USB 3.2 Gen2", "rugged": True}, "IP65 rugged portable SSD for footage."),
    ("Storage", "external ssd", "SanDisk", "Extreme Portable SSD 1TB", 109, {"capacity_gb": 1000, "interface": "USB 3.2 Gen2"}, "Carabiner loop, 1050 MB/s."),
    ("Storage", "external hdd", "Seagate", "Expansion 4TB External HDD", 99, {"capacity_gb": 4000, "interface": "USB 3.0"}, "Large backup drive."),
    ("Storage", "external hdd", "WD", "My Passport 2TB", 79, {"capacity_gb": 2000, "interface": "USB 3.0", "encryption": "AES-256"}, "Password protection with hardware encryption."),
    ("Memory", "laptop ram", "Crucial", "16GB DDR5-5600 SODIMM", 49, {"capacity_gb": 16, "type": "DDR5 SODIMM"}, "Laptop memory upgrade."),
    ("Memory", "laptop ram", "Kingston", "FURY Impact 32GB (2x16) DDR5 SODIMM", 109, {"capacity_gb": 32, "type": "DDR5 SODIMM"}, "Plug-and-play overclocking for laptops."),
    ("Memory", "desktop ram", "Corsair", "Vengeance RGB 32GB (2x16) DDR5-6000", 119, {"capacity_gb": 32, "type": "DDR5 DIMM"}, "Intel XMP 3.0 / AMD EXPO."),
    ("Memory", "desktop ram", "G.Skill", "Trident Z5 64GB (2x32) DDR5-6400", 229, {"capacity_gb": 64, "type": "DDR5 DIMM"}, "High-capacity kit for creators."),
    ("Graphics Cards", "nvidia", "ASUS", "Dual GeForce RTX 4060 8GB", 299, {"vram_gb": 8}, "1080p ultra gaming with DLSS 3."),
    ("Graphics Cards", "nvidia", "MSI", "GeForce RTX 4070 SUPER Ventus 12GB", 599, {"vram_gb": 12}, "1440p high-refresh gaming."),
    ("Graphics Cards", "nvidia", "Gigabyte", "GeForce RTX 4080 SUPER Gaming OC 16GB", 1049, {"vram_gb": 16}, "4K gaming and AI workloads."),
    ("Graphics Cards", "amd", "Sapphire", "Pulse Radeon RX 7800 XT 16GB", 499, {"vram_gb": 16}, "Excellent 1440p value."),
    ("Networking", "router", "TP-Link", "Archer AX55 Wi-Fi 6 Router", 99, {"wifi": "Wi-Fi 6", "speed": "AX3000"}, "Dual-band Wi-Fi 6 for home offices."),
    ("Networking", "mesh", "TP-Link", "Deco X50 Mesh (3-pack)", 249, {"wifi": "Wi-Fi 6"}, "Whole-home mesh coverage."),
    ("Networking", "router", "ASUS", "ROG Rapture GT-AXE11000", 449, {"wifi": "Wi-Fi 6E"}, "Tri-band gaming router."),
    ("Networking", "adapter", "TP-Link", "UE300 USB 3.0 to Gigabit Ethernet", 15, {}, "Wired Ethernet for thin laptops."),
    ("Bags & Sleeves", "backpack", "Targus", "15.6\" Classic Backpack", 39, {"fits_in": 15.6}, "Padded laptop compartment."),
    ("Bags & Sleeves", "backpack", "Samsonite", "Pro-DLX 6 Backpack 17\"", 159, {"fits_in": 17}, "Business travel backpack with RFID pocket."),
    ("Bags & Sleeves", "sleeve", "tomtoc", "360 Protective Sleeve 14\"", 29, {"fits_in": 14}, "CornerArmor drop protection."),
    ("Bags & Sleeves", "sleeve", "Incase", "Compact Sleeve 13\" MacBook", 49, {"fits_in": 13}, "Slim sleeve for MacBook Air/Pro."),
    ("Chargers & Power", "charger", "Anker", "Prime 100W GaN USB-C Charger", 69, {"watts": 100, "ports": 3}, "Charge a laptop + phone + tablet."),
    ("Chargers & Power", "charger", "Apple", "70W USB-C Power Adapter", 59, {"watts": 70}, "Official MacBook charger."),
    ("Chargers & Power", "power bank", "Anker", "737 Power Bank 24,000mAh 140W", 109, {"mah": 24000, "watts": 140}, "Can charge most USB-C laptops."),
    ("Chargers & Power", "ups", "APC", "Back-UPS 650VA", 89, {"va": 650}, "Protects desktops from power cuts."),
    ("Cooling Pads", "cooling", "Cooler Master", "NotePal X3", 39, {"fans": 1, "fits_in": 17}, "200mm fan cooling pad for gaming laptops."),
    ("Cooling Pads", "cooling", "KLIM", "Ultimate RGB Cooling Pad", 35, {"fans": 5}, "Five fans, adjustable height."),
    ("Desktops", "mini pc", "Apple", "Mac mini (M4) 16GB/256GB", 599, {"ram_gb": 16, "storage_gb": 256, "cpu": "Apple M4"}, "Tiny, silent and powerful desktop."),
    ("Desktops", "mini pc", "Intel", "NUC 13 Pro Mini PC i7", 749, {"ram_gb": 16, "storage_gb": 512, "cpu": "Intel Core i7-1360P"}, "Compact office PC with dual 4K output."),
    ("Desktops", "all-in-one", "HP", "Pavilion All-in-One 27\"", 999, {"ram_gb": 16, "storage_gb": 1024, "cpu": "Intel Core i7-1355U"}, "Family PC with built-in webcam."),
    ("Desktops", "all-in-one", "Apple", "iMac 24\" (M4)", 1299, {"ram_gb": 16, "storage_gb": 256, "cpu": "Apple M4"}, "4.5K Retina all-in-one."),
    ("Desktops", "gaming desktop", "Lenovo", "Legion Tower 5i Gen 8 RTX 4070", 1599, {"ram_gb": 32, "storage_gb": 1024, "gpu": "RTX 4070"}, "Tool-less upgradeable gaming tower."),
    ("Desktops", "gaming desktop", "HP", "OMEN 25L RTX 4060", 1199, {"ram_gb": 16, "storage_gb": 1024, "gpu": "RTX 4060"}, "Entry high-refresh 1080p gaming."),
    ("Desktops", "business desktop", "Dell", "OptiPlex 7020 SFF", 849, {"ram_gb": 16, "storage_gb": 512, "cpu": "Intel Core i5-14500"}, "Small form factor business desktop."),
]


def _sku(prefix: str, n: int) -> str:
    return f"{prefix}-{n:05d}"


def offline_catalog(seed: int = 7) -> list[dict]:
    """Deterministic catalog of ~190 SKUs with realistic specs, priced in SAR."""
    rng = random.Random(seed)
    out: list[dict] = []
    n = 0
    for brand, model, sub, base in LAPTOP_LINES:
        configs = rng.sample(LAPTOP_CONFIGS, k=3 if sub in ("gaming", "creator", "business") else 2)
        for ram, ssd, bump in sorted(configs):
            if sub in ("gaming", "creator") and ram < 16:
                ram, bump = 16, bump + 120
            n += 1
            cpu = "Apple " + model.split("(")[-1].rstrip(")") if brand == "Apple" else rng.choice(CPU_BY_SUB[sub])
            if re.search(r"\b(A15|G14|TUF)\b", model):
                cpu = rng.choice(["AMD Ryzen 7 8845HS", "AMD Ryzen 9 8945HS"])
            if brand == "Apple" or sub not in GPU_BY_SUB:
                gpu = "Integrated GPU" if brand == "Apple" else "Integrated graphics"
            else:  # GPU tier follows the price tier of the line (no RTX 4080 in a budget chassis)
                tier = ["NVIDIA GeForce RTX 4050 6GB", "NVIDIA GeForce RTX 4060 8GB"] if base < 1000 else \
                    ["NVIDIA GeForce RTX 4060 8GB", "NVIDIA GeForce RTX 4070 8GB"] if base < 1700 else \
                    ["NVIDIA GeForce RTX 4070 8GB", "NVIDIA GeForce RTX 4080 12GB"]
                gpu = tier[min(1, bump // 300)]
                bump += 0 if gpu.endswith(("4050 6GB", "4060 8GB")) else 250
            screen = 16.0 if "16" in model else 15.6 if "15" in model else 14.0 if "14" in model else 13.6 if "13" in model else 14.0
            refresh = rng.choice([144, 165, 240]) if sub == "gaming" else rng.choice([60, 60, 120])
            weight = round(rng.uniform(2.1, 2.7), 2) if sub == "gaming" else round(rng.uniform(1.15, 1.75), 2)
            battery = rng.randint(5, 8) if sub == "gaming" else rng.randint(10, 20)
            usd = base + bump + rng.choice([-30, 0, 20, 50])
            specs = {"cpu": cpu, "ram_gb": ram, "storage_gb": ssd, "gpu": gpu, "screen_in": screen,
                     "refresh_hz": refresh, "weight_kg": weight, "battery_h": battery,
                     "os": "macOS" if brand == "Apple" else "Windows 11 Pro" if sub == "business" else "Windows 11 Home"}
            title = f"{brand} {model} - {cpu.replace('Intel ', '').replace('AMD ', '')}, {ram}GB RAM, " \
                    f"{ssd if ssd < 1000 else ssd // 1024}{'GB' if ssd < 1000 else 'TB'} SSD"
            if gpu.startswith("NVIDIA"):
                title += f", {gpu.replace('NVIDIA GeForce ', '')}"
            features = [
                f"{screen}\" {'high-refresh ' if refresh > 60 else ''}{refresh}Hz display",
                f"{cpu} processor", f"{ram}GB memory, {ssd}GB NVMe SSD", f"{gpu}",
                f"Up to {battery} hours battery, {weight} kg",
            ]
            if sub == "business":
                features.append("TPM 2.0, fingerprint reader, IR camera, MIL-STD-810H tested")
            out.append(_product(f"LAP{n:04d}", _sku("LAP", n), title, brand, "Laptops", sub, usd, rng, specs,
                                features, f"{USE_CASE_COPY[sub]} {brand} {model} configured with {ram}GB RAM and "
                                f"{ssd}GB SSD."))
    for cat, sub, brand, model, base, spec, blurb in ACCESSORY_LINES:
        n += 1
        variants = [("", 0)]
        if cat in ("Mice", "Keyboards", "Headsets") and rng.random() < 0.4:
            variants.append((" (White)", 5))
        for suffix, bump in variants:
            pid = f"ACC{n:04d}{'W' if suffix else ''}"
            features = [blurb] + [f"{k.replace('_', ' ')}: {v}" for k, v in spec.items()]
            out.append(_product(pid, _sku(cat[:3].upper(), n), f"{brand} {model}{suffix}", brand, cat, sub,
                                base + bump, rng, dict(spec), features,
                                f"{brand} {model}. {blurb} A popular {sub} choice in our {cat.lower()} range."))
    return out


def _product(pid, sku, title, brand, cat, sub, usd, rng, specs, features, desc, source="offline") -> dict:
    price = round(usd * USD_TO_SAR / 5) * 5 - 1  # retail-style 4,499 pricing
    return {
        "product_id": pid, "sku": sku, "title": title, "brand": brand, "category": cat, "subcategory": sub,
        "price": float(price), "cost": round(price * rng.uniform(0.72, 0.86), 2),
        "rating": round(min(5.0, rng.gauss(4.35, 0.3)), 1), "rating_count": int(rng.lognormvariate(5, 1)),
        "specs_json": json.dumps(specs), "features": "\n".join(features), "description": desc, "source": source,
    }


# --------------------------------------------------------------------------------------
# Hugging Face / UCSD loader
# --------------------------------------------------------------------------------------
HF_DATASET = "McAuley-Lab/Amazon-Reviews-2023"
HF_CONFIG = "raw_meta_Electronics"
UCSD_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz"

_CAT_RULES = [
    ("Laptops", r"\blaptops?\b|notebook|chromebook|macbook"), ("Desktops", r"\bdesktops?\b|mini pc|all-in-one"),
    ("Monitors", r"\bmonitors?\b"), ("Keyboards", r"keyboard"), ("Mice", r"\bmice\b|\bmouse\b"),
    ("Headsets", r"headset|headphone"), ("Webcams", r"webcam"), ("Docks & Hubs", r"dock|usb hubs?|\bhub\b"),
    ("Storage", r"\bssd\b|solid state|hard drive|external drive|flash drive"), ("Memory", r"\bram\b|memory|dimm"),
    ("Graphics Cards", r"graphics card|\bgpu\b|geforce|radeon"), ("Networking", r"router|networking|wi-?fi|ethernet"),
    ("Bags & Sleeves", r"\bbags?\b|backpack|sleeve|case"), ("Chargers & Power", r"charger|power bank|adapter|ups"),
    ("Cooling Pads", r"cooling pad|cooler"),
]


def _map_category(rec: dict) -> str | None:
    text = " ".join((rec.get("categories") or [])[1:] + [rec.get("title") or ""]).lower()
    for cat, pat in _CAT_RULES:
        if re.search(pat, text):
            return cat
    return None


def _parse_price(p) -> float | None:
    if p is None:
        return None
    try:
        return float(str(p).replace("$", "").replace(",", "").split()[0])
    except (ValueError, IndexError):
        return None


def _from_amazon(rec: dict, rng: random.Random) -> dict | None:
    cats = rec.get("categories") or []
    if "Computers & Accessories" not in cats:
        return None
    cat, usd = _map_category(rec), _parse_price(rec.get("price"))
    title = (rec.get("title") or "").strip()
    if not cat or not usd or not title or usd < 5 or usd > 6000:
        return None
    details = rec.get("details") or {}
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = {}
    brand = (details.get("Brand") or rec.get("store") or title.split()[0]).strip()
    features = [f for f in (rec.get("features") or []) if f][:8]
    desc = " ".join(d for d in (rec.get("description") or []) if d)[:1500] or (features[0] if features else title)
    p = _product(f"AMZ{rec['parent_asin']}", rec["parent_asin"], title[:180], brand, cat,
                 (cats[-1] if len(cats) > 2 else cat).lower(), usd, rng,
                 {k: str(v)[:80] for k, v in list(details.items())[:12]}, features, desc, source="amazon2023")
    if rec.get("average_rating"):
        p["rating"] = float(rec["average_rating"])
        p["rating_count"] = int(rec.get("rating_number") or 0)
    return p


def _iter_hf() -> Iterable[dict]:
    from datasets import load_dataset  # optional dependency

    ds = load_dataset(HF_DATASET, HF_CONFIG, split="full", streaming=True)
    yield from ds


def _iter_ucsd(max_lines: int) -> Iterable[dict]:
    import urllib.request

    with urllib.request.urlopen(UCSD_URL, timeout=30) as resp:  # noqa: S310 - fixed public URL
        with gzip.GzipFile(fileobj=io.BufferedReader(resp)) as gz:
            for i, line in enumerate(gz):
                if i >= max_lines:
                    break
                yield json.loads(line)


def load_catalog(source: str = "auto", limit: int = 600, scan_limit: int = 400_000, seed: int = 7) -> list[dict]:
    """Return a product list. source: auto | hf | ucsd | offline."""
    if source == "offline":
        return offline_catalog(seed)
    rng = random.Random(seed)
    attempts = ["hf", "ucsd"] if source == "auto" else [source]
    for src in attempts:
        try:
            it = _iter_hf() if src == "hf" else _iter_ucsd(scan_limit)
            per_cat: dict[str, int] = {}
            out: list[dict] = []
            for i, rec in enumerate(it):
                if i >= scan_limit or len(out) >= limit:
                    break
                p = _from_amazon(rec, rng)
                if p and per_cat.get(p["category"], 0) < max(12, limit // len(CATEGORIES)):
                    per_cat[p["category"]] = per_cat.get(p["category"], 0) + 1
                    out.append(p)
            if len(out) >= 50:
                log.info("Loaded %d products from %s", len(out), src)
                # keep the curated laptop range too: Amazon meta is accessory-heavy
                return out + [p for p in offline_catalog(seed) if p["category"] in ("Laptops", "Desktops")]
            log.warning("%s returned only %d usable products; trying next source", src, len(out))
        except Exception as exc:  # network / optional dep missing
            log.warning("Catalog source %s unavailable (%s)", src, exc)
    log.warning("Falling back to offline catalog")
    return offline_catalog(seed)
