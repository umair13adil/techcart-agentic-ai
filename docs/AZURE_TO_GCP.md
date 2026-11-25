# Azure implementation ↔ Google Cloud (Vertex AI / Gemini Enterprise Agent Platform)

The project runs on Azure OpenAI because those are the credits available, and every component sits behind a small seam (`llm.py`, `rag/embeddings.py`, `rag/store.py`, `ml/tracking.py`, `agents/mcp_bridge.py`). This table is the interview answer to "how would you build this on GCP?"

> Naming note (as of Nov 2026): in April 2026 Google rebranded and extended Vertex AI as **Gemini Enterprise Agent Platform**. Vertex AI services (Pipelines, Model Registry, Endpoints, Model Monitoring, Vector Search, RAG Engine) are still referred to by those names in the docs. Current Gemini models include Gemini 3.1 Pro and Gemini 3 Flash. Dialogflow CX now appears as **Conversational Agents (Dialogflow CX)**, which combines generative *playbooks*, data-store RAG and deterministic *flows*.

| Concern | This repo (Azure / OSS) | GCP equivalent | Migration notes |
|---|---|---|---|
| Chat LLM | Azure OpenAI deployment via `AzureChatOpenAI` | **Gemini 3 Flash** (routing, extraction) / **Gemini 3.1 Pro** (complex reasoning) via `ChatVertexAI` or the Gen AI SDK | Same LangChain interface. Use `with_structured_output`, which maps to Gemini's `response_schema` |
| Embeddings | `text-embedding-3-small` | Vertex `text-embedding` / `gemini-embedding` models | Re-embed the corpus. The content-hash cache makes it incremental |
| Vector + keyword search | Local BM25 + numpy + RRF, or **Azure AI Search** hybrid | **Vertex AI Vector Search** (hybrid dense + sparse), **Vertex AI Search** (managed RAG), or **RAG Engine** | Keep the `audience` filter as a restrict/namespace on every query. Security trimming must stay inside retrieval |
| Agent framework | LangGraph (StateGraph, interrupt, checkpointer) | LangGraph deployed on **Agent Engine**, or **ADK** (multi-agent, MCP-native) | ADK supports MCP tools, so the MCP servers here plug in unchanged |
| Conversational front door | FastAPI `/api/chat` + web widget | **Conversational Agents (Dialogflow CX)**: flows for deterministic checkout and returns, playbooks for open shopping chat, data stores for policy FAQ, a **webhook** that calls this API's search, recs and checkout endpoints | Map intents to flows (`order_status`, `return_refund`) and the shopper to a playbook. Deterministic flows suit payment steps |
| Classical models | scikit-learn + local registry | **Vertex AI Training** (custom container) → **Model Registry** (versions + aliases) → **Endpoints** (online) / Batch Prediction | `Registry.promote()` ≈ setting the `default` alias after evaluation passes |
| Pipelines / CT | `scripts/train_all.py` + GitHub Actions schedule | **Vertex AI Pipelines** (KFP v2 components: ingest → train → evaluate → conditional register/deploy), Cloud Scheduler trigger | Each script stage becomes a `@dsl.component`; gates become `dsl.If` on evaluation outputs |
| Experiment tracking | MLflow hook (Azure ML is MLflow-native) | **Vertex AI Experiments** / Vertex ML Metadata (or MLflow on GCP) | |
| Drift monitoring | PSI job (`ml/drift.py`) + exit code to trigger retrain | **Vertex AI Model Monitoring** (v1 GA on endpoints: feature skew/drift; v2 preview: model-version monitors, tabular) | Keep the custom PSI for LLM-side signals (intent mix, fallback rate), which tabular monitoring doesn't cover |
| Feature store | Features computed on the fly (`fraud.build_features`) | **Vertex AI Feature Store** (BigQuery-backed) for online fraud features | Prevents training/serving skew for `prior_orders` and `account_age` |
| Explainability | Rule reasons + feature list | **Vertex Explainable AI** (sampled Shapley / integrated gradients) | Attribution drift can also be monitored |
| Fine-tuning | Azure OpenAI fine-tuning (`scripts/finetune_intent_aoai.py`) | **Gemini supervised tuning** on Vertex AI | Same JSONL chat format idea; evaluate against the classical tier before paying for it |
| LLM evals / tracing | LangSmith | LangSmith still works, plus **Gen AI evaluation service** on Vertex | |
| Safety | Regex guardrails, PII redaction | **Model Armor**, **Sensitive Data Protection (DLP)** for PII de-identification | DLP de-identify before logs and BigQuery |
| Data | SQLite + docs folder | **BigQuery** (analytics, dashboard), Cloud SQL/AlloyDB (OLTP), GCS (docs) | Admin dashboard queries move to BigQuery views |
| Serving | FastAPI container | **Cloud Run** (API + MCP servers as separate services), GKE for heavier loads | docker-compose maps 1:1 to four Cloud Run services |
| Residency (PDPL) | Choose a KSA/GCC Azure region where available | **Dammam (me-central2)** region, data residency controls, CMEK | See RESPONSIBLE_AI.md |

## A two-minute verbal version

"The architecture is cloud-agnostic by design. On GCP I would put Gemini 3 Flash behind the same LangChain interface for routing and extraction, and use Vertex AI Vector Search with hybrid sparse+dense retrieval and the same audience restricts for RAG. The classical models train in Vertex AI Pipelines, where each of my scripts becomes a KFP component and the promotion gate becomes a conditional step that registers the model in Model Registry and deploys it to an endpoint. Model Monitoring covers feature drift on the fraud endpoint, and I keep my PSI job for LLM-specific signals. The customer-facing channel can be a Conversational Agents (Dialogflow CX) agent: deterministic flows for checkout and returns, a generative playbook for shopping, and a webhook into the same APIs. The MCP servers don't change, because ADK and Agent Engine speak MCP."
