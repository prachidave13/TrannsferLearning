"""Build / update logic.

Three modes:
- 'algorithmic' (default): TF-IDF + TextRank + MMR. No LLM, no API key.
- 'llm':                  uses OpenAI/Anthropic for fluent generation.
- 'skeleton':             empty skeleton + raw regex hints (legacy offline).
"""
from __future__ import annotations

from typing import Optional

from .extract import extract
from .llm import LLMUnavailable, call_llm
from .render import build_algorithmic
from .schema import (
    CAPSULE_SKELETON,
    SELF_SUMMARIZE_PROMPT,
    SYSTEM_PROMPT,
    UPDATE_SYSTEM_PROMPT,
)


def build_capsule(
    transcript: str,
    *,
    mode: str = "algorithmic",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Produce a capsule from a single transcript.

    mode:
      'algorithmic' — TextRank/MMR pipeline (default, no API)
      'llm'         — call OpenAI/Anthropic
      'skeleton'    — empty schema + regex hints
    """
    if mode == "algorithmic":
        return build_algorithmic(transcript)
    if mode == "skeleton":
        return _skeleton_capsule(transcript)
    if mode == "llm":
        hints = extract(transcript).to_hint_block()
        user_msg = (
            f"{hints}\n\n"
            "<TRANSCRIPT>\n"
            f"{transcript}\n"
            "</TRANSCRIPT>\n\n"
            "Produce the Context Capsule now."
        )
        try:
            return call_llm(SYSTEM_PROMPT, user_msg, provider=provider, model=model).strip()
        except LLMUnavailable as exc:
            return _skeleton_capsule(transcript, note=str(exc))
    raise ValueError(f"unknown mode: {mode!r}")


def update_capsule(
    old_capsule: str,
    new_transcript: str,
    *,
    mode: str = "algorithmic",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Merge new chat info into an existing capsule.

    Algorithmic update strategy: build a fresh capsule from the new
    transcript, then append it as a 'Session N' addendum to the old one,
    deduplicating bullets that are textually identical to the previous run.
    Decisions and dead ends ACCUMULATE — they are never dropped.
    """
    if mode == "algorithmic":
        new_capsule = build_algorithmic(new_transcript)
        return _append_session(old_capsule, new_capsule)
    if mode == "skeleton":
        hints = extract(new_transcript).to_hint_block()
        return (
            old_capsule.rstrip()
            + "\n\n---\n\n## Session Addendum (skeleton mode)\n\n"
            + hints
            + "\n"
        )
    if mode == "llm":
        user_msg = (
            "<EXISTING_CAPSULE>\n"
            f"{old_capsule}\n"
            "</EXISTING_CAPSULE>\n\n"
            "<NEW_TRANSCRIPT>\n"
            f"{new_transcript}\n"
            "</NEW_TRANSCRIPT>\n\n"
            "Produce the updated capsule."
        )
        try:
            return call_llm(
                UPDATE_SYSTEM_PROMPT, user_msg, provider=provider, model=model
            ).strip()
        except LLMUnavailable:
            return update_capsule(old_capsule, new_transcript, mode="algorithmic")
    raise ValueError(f"unknown mode: {mode!r}")


def _append_session(old: str, new: str) -> str:
    """Diff-aware append: only show bullets in `new` that aren't in `old`."""
    old_bullets = {ln.strip() for ln in old.splitlines() if ln.strip().startswith("- ")}
    fresh_lines = []
    for ln in new.splitlines():
        if ln.strip().startswith("- ") and ln.strip() in old_bullets:
            continue
        fresh_lines.append(ln)
    addendum = "\n".join(fresh_lines).strip()
    return (
        old.rstrip()
        + "\n\n---\n\n## Session Addendum (algorithmic, deduped)\n\n"
        + addendum
        + "\n"
    )


def _skeleton_capsule(transcript: str, note: str = "") -> str:
    """Empty skeleton + extracted hints + a self-summarize prompt."""
    hints = extract(transcript).to_hint_block()
    banner = "## Skeleton mode\n"
    if note:
        banner += f"_{note}_\n\n"
    banner += (
        "No LLM call was made. Below is a skeleton with heuristically-extracted "
        "hints. To get a real capsule, either:\n"
        "1. Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and re-run, **or**\n"
        "2. Paste the prompt at the bottom of this file into the original chat.\n"
    )
    return (
        banner
        + "\n---\n\n"
        + CAPSULE_SKELETON
        + "\n---\n\n## Extracted hints (raw)\n\n"
        + hints
        + "\n\n---\n\n## Self-summarize prompt (paste into original chat)\n\n"
        + "```\n"
        + SELF_SUMMARIZE_PROMPT
        + "\n```\n"
    )


def make_resume_prompt(capsule_path: str, capsule_body: str = "") -> str:
    """Ready-to-paste prompt for a fresh chat. Inlines the capsule body if given."""
    body = capsule_body.strip() or "<paste the contents of the capsule file here>"
    return (
        "I'm continuing work from a previous LLM session. The Context Capsule "
        "below is a structured handoff of everything we've decided, tried, and "
        "ruled out. Read it carefully — especially sections 3 (Decision Log) "
        "and 4 (Dead Ends) — before proposing anything. Do not re-suggest "
        "approaches listed in Dead Ends. After you've read it, acknowledge "
        "with a one-line summary of where we are and ask me for the next step.\n\n"
        f"--- BEGIN CONTEXT CAPSULE ({capsule_path}) ---\n\n"
        f"{body}\n\n"
        "--- END CONTEXT CAPSULE ---\n"
    )
