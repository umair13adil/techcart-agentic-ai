"""Smoke-test the Azure setup in .env before running the app.

    python scripts/check_azure.py      (or: make check-azure)

Checks, in order: endpoint DNS, embeddings deployment, chat deployment (with a tool call, which the agents need),
and the Entra ID app settings. Each failure prints what to fix in the Azure portal.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from techcart.config import settings  # noqa: E402
from techcart.llm import chat_model, check_endpoint, embeddings_client  # noqa: E402

ok = True


def step(name, fn, hint):
    global ok
    try:
        detail = fn()
        print(f"  OK    {name}{': ' + detail if detail else ''}")
    except Exception as e:  # noqa: BLE001 - report every failure, keep going
        ok = False
        print(f"  FAIL  {name}: {str(e).splitlines()[0][:300]}\n        -> {hint}")


def emb():
    v = embeddings_client().embeddings.create(model=settings.aoai_embed_deployment, input=["hello"])
    return f"{settings.aoai_embed_deployment}, {len(v.data[0].embedding)} dims"


def chat():
    from langchain_core.tools import tool

    @tool
    def ping(x: str) -> str:
        """Echo x."""
        return x

    m = chat_model().bind_tools([ping])
    r = m.invoke("Call the ping tool with x='techcart'.")
    if not r.tool_calls:
        raise RuntimeError("model answered but did not call the tool")
    return f"{settings.aoai_chat_deployment}, tool call OK"


def entra():
    missing = [k for k, v in {"ENTRA_CLIENT_ID": settings.entra_client_id,
                               "ENTRA_CLIENT_SECRET": settings.entra_client_secret,
                               "ADMIN_EMAILS": ",".join(settings.admin_emails)}.items() if not v]
    if missing:
        raise RuntimeError("missing " + ", ".join(missing))
    return f"tenant={settings.entra_tenant}, redirect={settings.entra_redirect_uri}"


print("Azure setup check")
if not settings.llm_enabled:
    print("  FAIL  AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY not set in .env")
    ok = False
else:
    step("endpoint reachable", lambda: (check_endpoint(), settings.aoai_endpoint)[1],
         "copy the endpoint from Azure portal -> Foundry resource -> Keys and Endpoint")
    step("embeddings deployment", emb,
         "deployment name must match AZURE_OPENAI_EMBED_DEPLOYMENT (404 = wrong name, 401 = wrong key)")
    step("chat deployment", chat,
         "deployment name must match AZURE_OPENAI_CHAT_DEPLOYMENT; name it after the model (e.g. gpt-5.4-mini); "
         "API version >= 2025-04-01-preview")
step("Entra ID sign-in settings", entra, "see docs/AZURE_SETUP.md, part 3")
print("All good - run: make bootstrap && make api" if ok else "\nFix the FAIL lines above, then re-run.")
sys.exit(0 if ok else 1)
