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

    check_endpoint()
    kwargs = dict(azure_endpoint=settings.aoai_endpoint, api_key=settings.aoai_key,
                  api_version=settings.aoai_api_version, azure_deployment=settings.aoai_chat_deployment,
                  timeout=30, max_retries=2)
    # reasoning-family deployments (gpt-5.x, o-series) reject a custom temperature
    if not settings.aoai_chat_deployment.startswith(("gpt-5", "o")):
        kwargs["temperature"] = temperature
    return AzureChatOpenAI(**kwargs)


def check_endpoint() -> None:
    """Fail fast with a readable message when the Azure endpoint can't be reached (typo, placeholder, quotes)."""
    import socket
    from urllib.parse import urlparse

    url = settings.aoai_endpoint
    host = urlparse(url).hostname
    if not url.startswith("https://") or not host or "<" in url:
        raise LLMUnavailable(f"AZURE_OPENAI_ENDPOINT looks wrong: {url!r}. Expected "
                             "https://<resource>.openai.azure.com/ or https://<resource>.cognitiveservices.azure.com/")
    try:
        socket.getaddrinfo(host, 443)
    except OSError as e:
        raise LLMUnavailable(f"Cannot resolve {host!r} ({e}). Check the resource name in AZURE_OPENAI_ENDPOINT "
                             "(Azure portal -> your Azure OpenAI / Foundry resource -> Keys and Endpoint), "
                             "and your internet/VPN/DNS.") from None


@lru_cache(maxsize=1)
def embeddings_client():
    if not settings.llm_enabled:
        raise LLMUnavailable("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
    check_endpoint()
    from openai import AzureOpenAI

    return AzureOpenAI(azure_endpoint=settings.aoai_endpoint, api_key=settings.aoai_key,
                       api_version=settings.aoai_api_version)
