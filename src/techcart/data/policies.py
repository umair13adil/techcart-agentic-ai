"""Store policy / FAQ documents used for RAG. Fictional store ("TechCart"), KSA-based, prices in SAR."""

POLICIES: dict[str, tuple[str, str]] = {
    "policy-returns": ("Returns & Refunds Policy", """
TechCart customers may return most items within **7 days of delivery** for a full refund, provided the item is
unused, in its original packaging with all accessories, manuals and the original receipt or order number.
Exchanges (same item, different colour/size/config) are accepted within **14 days of delivery**.

Non-returnable items: opened software licences, items with broken anti-tamper seals on storage/memory, earbuds and
in-ear headsets once the hygiene seal is removed, and custom-configured (CTO) laptops.

Dead-on-arrival (DOA): if a product is faulty within 14 days of delivery we replace it at no cost after a quick
diagnostic (usually 48 hours at our Riyadh service centre).

Refund timing: mada and credit cards 5-10 business days; Apple Pay 3-7 business days; Tabby instalments are cancelled
and paid instalments are returned within 7 business days; Cash on Delivery orders are refunded by bank transfer (IBAN
required) within 7 business days.

Return shipping is free for DOA and wrong-item cases; otherwise a 25 SAR pickup fee applies. Start a return from
"My Orders" or ask the TechCart assistant with your order number.
"""),
    "policy-warranty": ("Warranty & TechCare+ Protection Plans", """
All products carry the **manufacturer warranty** (laptops and desktops typically 1 year, Apple 1 year, Lenovo
ThinkPad and Dell Latitude business models 3 years on-site where stated, monitors 3 years, accessories 1-2 years).

TechCart is the first point of contact for warranty claims in Saudi Arabia; we log the claim with the brand service
centre and keep you updated.

**TechCare+ extended protection** (purchase within 30 days of the device):
| Device price (SAR) | 1-year extension | 2-year extension | Accidental damage add-on |
|---|---|---|---|
| up to 2,000 | 149 | 249 | +99 |
| 2,001 - 5,000 | 299 | 499 | +199 |
| 5,001 - 10,000 | 499 | 849 | +349 |
| above 10,000 | 799 | 1,299 | +549 |

TechCare+ covers parts and labour, a loaner laptop for repairs longer than 5 business days (Riyadh, Jeddah, Dammam),
and one battery replacement if capacity drops below 80%. Liquid damage is covered only with the accidental damage
add-on. Claims are limited to two accidental-damage incidents per plan year.
"""),
    "policy-shipping": ("Shipping & Delivery", """
- **Riyadh**: next-day delivery for orders placed before 6 PM; same-day express (49 SAR) before 12 PM.
- **Jeddah, Dammam, Khobar, Mecca, Medina**: 1-3 business days.
- **Other KSA cities**: 2-5 business days.
- Shipping is **free on orders above 200 SAR**; otherwise 25 SAR. Express same-day is always 49 SAR.
- Large-format items (monitors 32" and above, desktops) ship with tracked white-glove courier at no extra cost.
- Delivery partners: SMSA Express, Aramex and our own TechCart Fleet in Riyadh.
- Tracking numbers are sent by SMS and email when the order ships; the assistant can also look up tracking.
- We currently do not ship outside Saudi Arabia.
"""),
    "policy-payments": ("Payment Options", """
Accepted payments: **mada**, Visa/Mastercard credit and debit cards, **Apple Pay**, **Tabby** (split into 4
interest-free payments for orders 100-5,000 SAR, subject to Tabby approval), and **Cash on Delivery** (COD) for orders
up to 2,500 SAR with a 15 SAR COD fee.

All prices include 15% VAT; a VAT invoice (ZATCA compliant, with QR code) is issued with every order.

For orders above 10,000 SAR or business customers, bank transfer is available after a quotation is accepted.
Approved business accounts may receive Net-30 payment terms. Card payments are 3-D Secure authenticated and
TechCart never stores full card numbers.
"""),
    "policy-b2b": ("Business & Bulk Purchase (B2B) Program", """
Companies, schools and government entities buying **10 or more units** can request a formal quotation.

- Volume discounts: 10-24 units 5%; 25-49 units 8%; 50+ units 12% (on laptops, desktops and monitors).
- Quotations are valid for **14 days** and lock price and stock allocation for that period.
- Optional services: asset tagging (15 SAR/unit), Windows Autopilot / Intune enrolment (25 SAR/unit), imaging with
  your corporate image (40 SAR/unit), and on-site deployment in Riyadh.
- Payment: bank transfer on acceptance, or Net-30 for approved accounts with a valid Commercial Registration (CR)
  and VAT number.
- Lead time for stocked items is 2-4 business days; non-stocked configurations 2-3 weeks from the distributor.
Ask the assistant for a quote or email b2b@techcart.example.
"""),
    "policy-pricematch": ("Price Match Promise", """
If you find the same new product (same model and configuration) cheaper at an authorised Saudi retailer within 7 days
of purchase, we refund the difference. Excludes marketplace sellers, clearance/open-box, bundles, and White Friday
door-buster deals. One claim per product.
"""),
    "policy-tradein": ("Trade-In Program", """
Trade in your old laptop, tablet or monitor for TechCart credit. Get an instant estimate from the assistant by
describing brand, model, age and condition. Typical values: laptops under 3 years old in good condition 15-35% of
original price; MacBooks retain 30-50%. Devices must power on, have no iCloud/BitLocker lock, and include a charger.
Credit is applied to your next order within 30 days. Data is wiped to NIST 800-88 standard and a wipe certificate is
emailed to you.
"""),
    "policy-privacy": ("Privacy Notice (PDPL)", """
TechCart processes personal data in line with the Saudi **Personal Data Protection Law (PDPL)** and its Implementing
Regulations.

- **What we collect**: name, phone, email, delivery address, order history, and chat/search interactions with the
  TechCart assistant. Payment card data is handled by our PCI-DSS certified payment processor and never stored by us.
- **Purposes**: fulfilling orders, customer support, fraud prevention, and - only with your consent - personalised
  recommendations and marketing.
- **Legal basis**: contract performance, legal obligations (VAT/ZATCA), legitimate interest (fraud prevention), and
  consent (marketing, personalisation). You can withdraw consent at any time in Account Settings.
- **Your rights**: to be informed, to access, to obtain a copy, to correct, and to request destruction of your data.
- **Retention**: order and invoice records 10 years (tax law); chat transcripts 12 months; search logs 90 days and
  pseudonymised.
- **AI assistant**: conversations are processed by AI models hosted in our cloud tenancy; personal identifiers are
  masked before being stored in logs or used for model improvement. Automated recommendations never make
  decisions with legal effect about you.
- **Transfers**: data is stored in KSA regions where available; any cross-border transfer follows PDPL transfer
  rules. Contact the DPO at privacy@techcart.example.
"""),
    "faq-compatibility": ("Buying Guide & Compatibility FAQ", """
**Which RAM fits my laptop?** Most current laptops use DDR5 SODIMM; many ultrabooks (MacBook, XPS 13, Zenbook,
Surface) have soldered memory and cannot be upgraded - buy the RAM you need up front. Desktop RAM (DIMM) does not
fit laptops.

**Can I charge my laptop with a USB-C charger?** Yes if the laptop supports USB-C Power Delivery. Match or exceed the
original wattage: ultrabooks 65W, 14" performance laptops 90-100W, gaming laptops usually need their barrel charger.

**Which SSD should I buy?** For laptops with an M.2 slot choose a PCIe 4.0 NVMe 2280 drive (e.g. WD Black SN850X,
Samsung 990 PRO, Crucial P3 Plus). For backups and video footage use a portable SSD (Samsung T7 Shield) or HDD.

**How much RAM do I need?** Office/study: 8-16GB. Programming and multitasking: 16GB. Gaming: 16-32GB. Video editing,
3D, local AI models: 32GB+.

**Gaming laptop GPU guide**: RTX 4050 - 1080p medium/high; RTX 4060 - 1080p ultra; RTX 4070 - 1440p high;
RTX 4080 - 1440p/4K ultra.

**Monitor for MacBook?** Choose a USB-C/Thunderbolt monitor with power delivery (Dell U2723QE, LG 32UN880) so one
cable charges the Mac and carries video. Base M-series MacBook Air supports one or two external displays depending
on the chip generation.

**Laptop bag size**: pick a bag rated at or above your screen size (a 15.6" bag fits 14" and 15.6" laptops).
"""),
}
