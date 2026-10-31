from app.config import env
from app.llm.base import LLMClient, LLMError


def get_llm(provider: str | None = None) -> LLMClient:
    provider = provider or env("LLM_PROVIDER", "anthropic")
    if provider == "anthropic":
        from app.llm.anthropic_client import AnthropicClient
        return AnthropicClient()
    if provider == "openai":
        from app.llm.openai_client import OpenAIClient
        return OpenAIClient()
    raise LLMError(f"Unknown LLM provider: {provider}")
