"""Azure OpenAI (Microsoft Foundry) model factories. Imported lazily so the classical-ML parts run without
LLM dependencies. LangSmith tracing is automatic when LANGSMITH_TRACING=true + LANGSMITH_API_KEY are set."""
from __future__ import annotations

from functools import lru_cache

from .config import settings


class LLMUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=4)
def chat_model(temperature: float = 0.0):
    if not settings.llm_enabled:
        raise LLMUnavailable("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
    from langchain_openai import AzureChatOpenAI

    kwargs = dict(azure_endpoint=settings.aoai_endpoint, api_key=settings.aoai_key,
                  api_version=settings.aoai_api_version, azure_deployment=settings.aoai_chat_deployment,
                  timeout=30, max_retries=2)
    # reasoning-family deployments (gpt-5.x, o-series) reject a custom temperature
    if not settings.aoai_chat_deployment.startswith(("gpt-5", "o")):
        kwargs["temperature"] = temperature
    return AzureChatOpenAI(**kwargs)


@lru_cache(maxsize=1)
def embeddings_client():
    if not settings.llm_enabled:
        raise LLMUnavailable("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
    from openai import AzureOpenAI

    return AzureOpenAI(azure_endpoint=settings.aoai_endpoint, api_key=settings.aoai_key,
                       api_version=settings.aoai_api_version)
