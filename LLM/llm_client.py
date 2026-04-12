"""
module5/llm_client.py
---------------------
Unified LLM client — supports Claude (Anthropic), OpenAI GPT, and a
local mock for testing. Configured via environment variables.

Usage:
    client = LLMClient()          # auto-detects provider from .env
    response = client.chat("Summarise these products: ...")
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()   # "anthropic" | "openai" | "mock"


class LLMClient:
    """
    Provider-agnostic LLM wrapper.
    Set LLM_PROVIDER in .env to switch between backends.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or PROVIDER).lower()
        self._client  = None
        self._init_client()

    def _init_client(self):
        if self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=os.environ["ANTHROPIC_API_KEY"]
                )
                self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
                logger.info(f"LLM: Anthropic / {self.model}")
            except (ImportError, KeyError) as e:
                logger.warning(f"Anthropic init failed ({e}) — falling back to mock")
                self.provider = "mock"

        elif self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                logger.info(f"LLM: OpenAI / {self.model}")
            except (ImportError, KeyError) as e:
                logger.warning(f"OpenAI init failed ({e}) — falling back to mock")
                self.provider = "mock"

        if self.provider == "mock":
            logger.info("LLM: Mock provider (set LLM_PROVIDER=anthropic or openai)")

    def chat(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        """Send a prompt and return the text response."""
        if self.provider == "anthropic":
            return self._call_anthropic(prompt, system, max_tokens)
        elif self.provider == "openai":
            return self._call_openai(prompt, system, max_tokens)
        else:
            return self._mock_response(prompt)

    def _call_anthropic(self, prompt: str, system: str, max_tokens: int) -> str:
        import anthropic
        msgs = [{"role": "user", "content": prompt}]
        kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": msgs}
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text

    def _call_openai(self, prompt: str, system: str, max_tokens: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )
        return resp.choices[0].message.content

    def _mock_response(self, prompt: str) -> str:
        """Deterministic mock — useful for tests without an API key."""
        if "summar" in prompt.lower():
            return "Compact summary: high-quality product with strong customer ratings and competitive pricing."
        if "trend" in prompt.lower() or "rapport" in prompt.lower():
            return ("Weekly trend report: Electronics leads with +12% score vs last week. "
                    "Discount products up 8%. Premium segment stable.")
        if "profil" in prompt.lower() or "client" in prompt.lower():
            return ("Target customer profile: tech-savvy consumers aged 25-40, "
                    "price-conscious but quality-driven, active online shoppers.")
        if "stratégie" in prompt.lower() or "marketing" in prompt.lower():
            return ("Recommended strategy: focus flash promotions on high-rated Electronics, "
                    "bundle Sport + Beauty products, and highlight customer reviews for Premium segment.")
        if "chatbot" in prompt.lower() or "?" in prompt:
            return "Based on the current data, the top-rated product this week is Product 42 with a score of 0.91."
        return f"[Mock LLM] Processed: {prompt[:80]}..."