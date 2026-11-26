PY ?= python
export PYTHONPATH := src

.PHONY: install bootstrap seed seed-refresh train index eval drift test api mcp-http lint docker

install:            ## install runtime + dev deps
	$(PY) -m pip install -e ".[dev]"

bootstrap: seed train index   ## store setup: real catalog, cold-start models, RAG index

seed:               ## real products from Hugging Face (cached in .cache/) + opening stock + suppliers; no fake activity
	$(PY) scripts/seed.py $(ARGS)

seed-refresh:       ## re-download the Hugging Face catalog
	$(PY) scripts/seed.py --refresh $(ARGS)

train:              ## train + evaluate + register + gated promotion
	$(PY) scripts/train_all.py

check-azure:        ## verify Foundry deployments + Entra settings from .env
	$(PY) scripts/check_azure.py

index:              ## chunk + embed docs (Azure OpenAI if configured, else local hashing)
	$(PY) scripts/build_index.py $(ARGS)

eval:               ## RAG retrieval eval (+ LangSmith LLM-judge with: make eval ARGS=--langsmith)
	$(PY) scripts/eval_rag.py $(ARGS)

drift:
	$(PY) scripts/monitor_drift.py

test:
	$(PY) -m pytest -q

api:                ## shop: http://localhost:8000  admin: http://localhost:8000/admin  docs: /docs
	$(PY) -m uvicorn techcart.api.main:app --reload --port 8000

mcp-http:           ## run the 3 integration MCP servers as separate HTTP services (then MCP_MODE=http make api)
	$(PY) -m techcart.mcp_servers.inventory_erp --http --port 8101 & \
	$(PY) -m techcart.mcp_servers.payments --http --port 8102 & \
	$(PY) -m techcart.mcp_servers.shipping --http --port 8103 & wait

lint:
	ruff check src tests scripts

docker:
	docker compose up --build
