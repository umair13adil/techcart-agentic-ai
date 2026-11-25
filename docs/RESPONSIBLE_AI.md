# Responsible AI & PDPL

Not legal advice. This file maps the engineering controls in this repo to the obligations a KSA retailer, insurer or healthcare provider would discuss with its DPO and legal counsel.

## Saudi PDPL, summarised for engineers
- Fully enforceable since **14 Sep 2024** and supervised by **SDAIA**. Implementing Regulations and Personal Data Transfer Regulations sit alongside the law.
- Extra-territorial: it applies to processing of personal data of people residing in KSA, even by entities outside KSA.
- Breach notification to SDAIA **within 72 hours** of becoming aware (where required). Data subjects are notified without undue delay.
- Cross-border transfers are allowed only under the Transfer Regulations' conditions and safeguards.
- Sensitive data includes health, genetic, biometric, credit, criminal and location data. This matters for the healthcare and insurance use cases in the JD.

## Controls implemented here

| Principle / obligation | Control in this repo | Where |
|---|---|---|
| Purpose limitation & consent | Personalised recommendations only with `marketing_consent`; otherwise trending items | `agents/tools.py::recommend_for_customer`, storefront banner |
| Data minimisation | PII regex redaction (phone, email, card, national ID/Iqama, IBAN) before writing search logs or audit logs | `guardrails.redact`, `services.audit/log_search` |
| Access control (need-to-know) | Retrieval-level security trimming by audience (`public` / `internal` / `owner`), plus row-level ownership for receipts. CI asserts zero leaks | `rag/retriever.py`, `rag/evaluate.py` |
| Authorisation | Order lookups require the order to belong to the caller | `services.order_status` |
| No card data in AI path | Payment method **tokens** only. The agent can't see or enter card numbers. 3-D Secure OTP is a human step | `mcp_servers/payments.py`, `agents/checkout.py` |
| Human oversight | Every money-moving action needs explicit human confirmation (`interrupt()`). Draft POs and quotes need staff approval | `agents/graph.py`, `inventory_erp.create_purchase_order` |
| Hard limits outside the prompt | Max basket total, qty per line, COD/Tabby rules enforced in code | `guardrails.validate_checkout` |
| Prompt-injection resistance | Heuristic filter, "context is data" system prompt, and tool output treated as untrusted | `guardrails.looks_like_injection`, `rag/retriever.SYSTEM_PROMPT` |
| Transparency / explainability | Search explains what it matched on and what it relaxed. Recs show "because you bought X". Fraud alerts list reasons. RAG answers cite sources | `search.py`, `ml/recommender.py`, `ml/fraud.reasons` |
| Auditability | Append-only agent audit trail (redacted), versioned model registry with metrics/params, training summary artifact | `agent_audit` table, `ml/tracking.py` |
| Retention | Policy states search logs 90 days and chats 12 months. Implement with a scheduled purge job | `data/policies.py` (policy-privacy) |
| No automated decisions with legal effect | Fraud model only **queues orders for human review**; it never auto-cancels | admin fraud queue |

## Fairness & bias checks to run before production
- **Fraud model**: compare alert rates and precision across city, account age band and payment method. Location and new-account features are legitimate risk signals, but they can act as proxies for nationality or income. Monitor *disparate alert rates*, not just overall precision.
- **Recommendations**: check exposure parity across brands and price tiers (popularity bias). Catalog coverage is reported as a metric.
- **Intent / LLM tier**: evaluate on Arabic, English and code-switched (Arabizi) queries separately. The hand-written test set includes Arabic for this reason.
- In insurance or healthcare settings: never use protected or sensitive attributes as model inputs without a documented legal basis. Run adverse-impact ratio (four-fifths rule) checks on approve/deny-style decisions. Prefer monotonic or interpretable models, or SHAP explanations, for underwriting and claims.
