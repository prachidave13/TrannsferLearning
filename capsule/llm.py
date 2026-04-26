"""LLM client — thin wrapper over OpenAI / Anthropic.

Picks a provider based on env vars unless one is forced.
"""
from __future__ import annotations

import os
from typing import Optional


class LLMUnavailable(RuntimeError):
    pass


def detect_provider(forced: Optional[str] = None) -> Optional[str]:
    if forced:
        return forced
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def call_llm(
    system: str,
    user: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Single completion call. Returns the assistant text."""
    provider = detect_provider(provider)
    if provider is None:
        raise LLMUnavailable(
            "No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY, "
            "or run with --offline."
        )

    if provider == "openai":
        return _call_openai(system, user, model or "gpt-4o-mini")
    if provider == "anthropic":
        return _call_anthropic(system, user, model or "claude-3-5-sonnet-latest")
    raise LLMUnavailable(f"Unknown provider: {provider}")


def _call_openai(system: str, user: str, model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMUnavailable(
            "openai package not installed. `pip install capsule[openai]`."
        ) from exc
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(system: str, user: str, model: str) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise LLMUnavailable(
            "anthropic package not installed. `pip install capsule[anthropic]`."
        ) from exc
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=0.2,
    )
    # Concatenate text blocks.
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
