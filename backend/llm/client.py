import os, json
from typing import Dict

from ..core.config import OPENAI_API_KEY

# Optional: if API key present, you could integrate a real LLM call using httpx.
# For this starter, we keep an interface and a fallback in explain.py.
def has_llm() -> bool:
    return bool(OPENAI_API_KEY)

async def call_llm(prompt: str) -> str:
    # Placeholder: integrate provider here.
    # Return JSON string with {"why": "..."} to match the contract.
    # In absence of a provider, raise to use fallback.
    raise RuntimeError("No LLM provider configured")
