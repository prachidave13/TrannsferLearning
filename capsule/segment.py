"""Sentence + turn segmentation.

Splits a chat transcript into a list of `Sentence` records, each tagged with:
- text          : the raw sentence
- turn_idx      : which turn it belongs to (0-indexed)
- role          : 'user' | 'assistant' | 'unknown'
- position      : normalized position in the chat in [0, 1]
- has_code      : whether this turn contained a code block
- in_code       : whether this sentence is inside a code block (then text is the block)
- code_lang     : if in_code, the fence language tag
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# Match a turn header like "**User:**", "User:", "Assistant:", "Human:" at line start.
_TURN_HEADER = re.compile(
    r"^\s*\*{0,2}(user|human|assistant|ai|system|me|you)\*{0,2}\s*:\s*\*{0,2}\s*",
    re.IGNORECASE | re.MULTILINE,
)

_CODE_FENCE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)

# Sentence terminator that ignores periods between digits (3.12) and after
# single-letter abbrevs.
_SENT_SPLIT = re.compile(r"(?<=[.!?])(?<!\d\.\d)\s+(?=[A-Z\"'`(\[])")


@dataclass
class Sentence:
    text: str
    turn_idx: int
    role: str
    position: float
    has_code: bool = False
    in_code: bool = False
    code_lang: str = ""
    # Filled later by feature stages:
    tags: List[str] = field(default_factory=list)
    score: float = 0.0


def _normalize_role(raw: str) -> str:
    r = raw.lower()
    if r in {"user", "human", "me"}:
        return "user"
    if r in {"assistant", "ai", "you"}:
        return "assistant"
    return "unknown"


def _split_turns(text: str):
    """Yield (role, body) pairs. If no headers found, treat whole text as one turn."""
    matches = list(_TURN_HEADER.finditer(text))
    if not matches:
        yield "unknown", text
        return
    for i, m in enumerate(matches):
        role = _normalize_role(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield role, text[start:end].strip()


def _clean(text: str) -> str:
    # Strip markdown emphasis/code-tick artifacts that aren't useful in bullets.
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_sentences(prose: str) -> List[str]:
    prose = prose.strip()
    if not prose:
        return []
    # Group lines into paragraphs/bullets:
    #   - A blank line ends a paragraph.
    #   - A bullet/number marker starts its own paragraph.
    #   - Otherwise consecutive lines are joined with a space (handles wrap).
    parts: List[str] = []
    buf: List[str] = []

    def flush():
        if buf:
            parts.append(" ".join(buf).strip())
            buf.clear()

    for line in prose.split("\n"):
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if re.match(r"^([-*+]|\d+\.)\s+", stripped):
            flush()
            parts.append(re.sub(r"^([-*+]|\d+\.)\s+", "", stripped))
        else:
            buf.append(stripped)
    flush()

    out: List[str] = []
    for p in parts:
        p = _clean(p)
        for s in _SENT_SPLIT.split(p):
            s = s.strip(" \t-—:")
            if len(s) >= 5:
                out.append(s)
    return out


def segment(transcript: str) -> List[Sentence]:
    sentences: List[Sentence] = []
    turns = list(_split_turns(transcript))
    n_turns = max(1, len(turns))
    for turn_idx, (role, body) in enumerate(turns):
        # Pull out code blocks first.
        code_spans = list(_CODE_FENCE.finditer(body))
        has_code = bool(code_spans)
        prose_parts = []
        last = 0
        code_records = []
        for m in code_spans:
            prose_parts.append(body[last:m.start()])
            code_records.append((m.group(1), m.group(2).strip()))
            last = m.end()
        prose_parts.append(body[last:])
        prose = " ".join(p.strip() for p in prose_parts if p.strip())

        # Sentences from prose.
        for s in _split_sentences(prose):
            pos = (turn_idx + 0.5) / n_turns
            sentences.append(
                Sentence(text=s, turn_idx=turn_idx, role=role, position=pos,
                         has_code=has_code)
            )
        # Each code block becomes one "sentence" record (treated specially).
        for lang, body_code in code_records:
            pos = (turn_idx + 0.5) / n_turns
            sentences.append(
                Sentence(text=body_code, turn_idx=turn_idx, role=role,
                         position=pos, has_code=True, in_code=True,
                         code_lang=lang)
            )
    return sentences
