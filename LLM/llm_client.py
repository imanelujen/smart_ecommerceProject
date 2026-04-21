"""
LLM/llm_client.py
---------------------
Unified LLM client — Anthropic, OpenAI, xAI Grok, **Groq** (OpenAI-compatible),
and mock. Loads `.env` from the project root and `LLM/` so Streamlit works from any CWD.

Usage:
    client = LLMClient()
    response = client.chat("...")
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LLM_DIR = Path(__file__).resolve().parent


def _load_env_files() -> None:
    """Load root `.env` then `LLM/.env` (later file overrides on duplicate keys)."""
    for path in (_REPO_ROOT / ".env", _LLM_DIR / ".env"):
        if path.is_file():
            load_dotenv(path, override=True)


_load_env_files()


def _infer_provider(explicit: Optional[str]) -> str:
    """
    Resolve which backend to use.

    - ``LLM_PROVIDER=mock`` → always mock (tests / offline).
    - ``LLM_PROVIDER=auto`` or unset → pick from available API keys.
    - If ``anthropic`` is set but there is no Anthropic key and a Groq-style
      key exists (``gsk_``), use **groq** (common misconfiguration).
    - ``GROK_API_KEY`` starting with ``gsk_`` is **Groq**, not xAI Grok.
    """
    raw = (explicit or os.getenv("LLM_PROVIDER", "auto")).strip().lower() or "auto"

    ant_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    oai_key = os.getenv("OPENAI_API_KEY", "").strip()
    grok_named = os.getenv("GROK_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    groq_material = groq_key or (grok_named.startswith("gsk_"))
    xai_material = grok_named.startswith("xai-")

    if raw == "mock":
        return "mock"

    if raw == "anthropic" and not ant_key:
        if groq_material:
            logger.warning(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is missing; "
                "using Groq (detected gsk_ key)."
            )
            return "groq"
        if xai_material:
            logger.warning(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is missing; "
                "using xAI Grok."
            )
            return "grok"

    if raw == "grok" and groq_material and not xai_material:
        logger.warning(
            "LLM_PROVIDER=grok but GROK_API_KEY looks like a Groq key (gsk_); "
            "using Groq API (api.groq.com)."
        )
        return "groq"

    if raw not in ("", "auto"):
        return raw

    if groq_material:
        return "groq"
    if xai_material:
        return "grok"
    if ant_key:
        return "anthropic"
    if oai_key:
        return "openai"
    return "mock"


PROVIDER = _infer_provider(None)


class LLMClient:
    """
    Provider-agnostic LLM wrapper.
    Set ``LLM_PROVIDER`` to ``anthropic`` | ``openai`` | ``grok`` | ``groq`` | ``mock`` | ``auto``.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = _infer_provider(provider)
        self._client = None
        self._init_client()

    def _init_client(self):
        api_key_anthropic = os.environ.get("ANTHROPIC_API_KEY", "")
        if self.provider == "anthropic" and api_key_anthropic.startswith("xai-"):
            self.provider = "grok"

        if self.provider == "anthropic":
            try:
                import anthropic  # type: ignore

                self._client = anthropic.Anthropic(api_key=api_key_anthropic)
                self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
                logger.info("LLM: Anthropic / %s", self.model)
            except (ImportError, KeyError) as e:
                logger.warning("Anthropic init failed (%s) — falling back to mock", e)
                self.provider = "mock"

        elif self.provider == "groq":
            try:
                from openai import OpenAI  # type: ignore

                api_key = (
                    os.getenv("GROQ_API_KEY", "").strip()
                    or os.getenv("GROK_API_KEY", "").strip()
                )
                if not api_key:
                    raise KeyError("GROQ_API_KEY or GROK_API_KEY (gsk_...) required for groq")
                self._client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
                self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
                logger.info("LLM: Groq / %s", self.model)
            except (ImportError, KeyError) as e:
                logger.warning("Groq init failed (%s) — falling back to mock", e)
                self.provider = "mock"

        elif self.provider in ("openai", "grok"):
            try:
                from openai import OpenAI  # type: ignore

                if self.provider == "grok":
                    api_key = os.getenv("GROK_API_KEY", api_key_anthropic).strip()
                    if api_key.startswith("gsk_"):
                        logger.warning(
                            "GROK_API_KEY is a Groq key; use LLM_PROVIDER=groq or GROQ_API_KEY."
                        )
                        raise KeyError("xAI key expected for grok provider")
                    self._client = OpenAI(
                        api_key=api_key, base_url="https://api.x.ai/v1"
                    )
                    self.model = os.getenv("GROK_MODEL", "grok-2-latest")
                    logger.info("LLM: Grok (xAI) / %s", self.model)
                else:
                    self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                    self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                    logger.info("LLM: OpenAI / %s", self.model)
            except (ImportError, KeyError) as e:
                logger.warning(
                    "%s init failed (%s) — falling back to mock",
                    self.provider.title(),
                    e,
                )
                self.provider = "mock"

        if self.provider == "mock":
            logger.info(
                "LLM: Mock provider — set LLM_PROVIDER=auto and a real API key, "
                "or install `openai` / `anthropic` for your provider."
            )

    def chat(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        """Send a prompt and return the text response."""
        if self.provider == "anthropic":
            return self._call_anthropic(prompt, system, max_tokens)
        if self.provider in ("openai", "grok", "groq"):
            return self._call_openai(prompt, system, max_tokens)
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
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0].message
        text = (choice.content or "").strip() if choice else ""
        return text

    def _mock_response(self, prompt: str) -> str:
        """Deterministic mock — useful for tests without an API key."""
        low = prompt.lower()
        if "summar" in low:
            return (
                "Compact summary: high-quality product with strong customer ratings "
                "and competitive pricing."
            )
        if "trend" in low or "rapport" in low:
            return (
                "Weekly trend report: Electronics leads with +12% score vs last week. "
                "Discount products up 8%. Premium segment stable."
            )
        if "profil" in low or "client" in low:
            return (
                "Target customer profile: tech-savvy consumers aged 25-40, "
                "price-conscious but quality-driven, active online shoppers."
            )
        if "stratégie" in low or "marketing" in low:
            return (
                "Recommended strategy: focus flash promotions on high-rated Electronics, "
                "bundle Sport + Beauty products, and highlight customer reviews for Premium segment."
            )
        if "chatbot" in low or "?" in prompt:
            return (
                "Based on the current data, the top-rated product this week is Product 42 "
                "with a score of 0.91."
            )
        return f"[Mock LLM] Processed: {prompt[:80]}..."
