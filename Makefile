PY ?= python
export PYTHONPATH := src

.PHONY: install bootstrap seed seed-hf train index traffic eval drift test api mcp-http lint docker

install:            ## install runtime + dev deps
	$(PY) -m pip install -e ".[dev]"

bootstrap: seed train index traffic   ## everything needed for a first run (offline-safe)

seed:               ## offline catalog (fast, deterministic)
	$(PY) scripts/seed.py --source offline

seed-hf:            ## real Amazon-Reviews-2023 catalog from Hugging Face (needs: pip install ".[hf]")
	$(PY) scripts/seed.py --source hf

train:              ## train + evaluate + register + gated promotion
	$(PY) scripts/train_all.py

index:              ## chunk + embed docs (Azure OpenAI if configured, else local hashing)
	$(PY) scripts/build_index.py

traffic:            ## backfill 30 days of search logs (drift + insights demo)
	$(PY) scripts/simulate_traffic.py

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
