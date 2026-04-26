"""Heuristic extractor — pulls high-signal bits from a raw chat transcript.

Used in two ways:
1. Offline mode: feed the user a partially-filled capsule + raw extracts.
2. LLM mode: prepend extracted hints to the transcript so the LLM doesn't
   miss them in a long context window.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# Regex bank. Conservative — false negatives are fine, false positives are not.
RE_CODE_FENCE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
RE_FILE_PATH = re.compile(
    r"(?<![\w/])([a-zA-Z0-9_\-./]+\.(?:py|js|ts|tsx|jsx|md|json|yml|yaml|toml|"
    r"go|rs|java|c|cpp|h|hpp|cs|rb|sh|sql|html|css|scss))\b"
)
RE_ERROR = re.compile(
    r"^(?:.*?(?:Error|Exception|Traceback|panic:|FAIL|failed|"
    r"undefined|cannot find|not found)[^\n]*)$",
    re.MULTILINE | re.IGNORECASE,
)
# Decision-ish phrasing.
# A "sentence terminator" excludes periods that sit between digits (e.g. 3.12)
# and periods inside paths (a.py is unlikely in prose anyway).
# We split on `.` followed by whitespace/EOL, or `\n`.
_SENT = r"(?:(?<!\d)\.(?=\s|$)|\n)"
_NON_TERM = r"(?:[^.\n]|(?<=\d)\.(?=\d))"  # any char except sentence terminator

RE_DECISION = re.compile(
    rf"({_NON_TERM}*\b(?:let'?s use|we'?ll go with|decided to|chose|going with|"
    rf"switching to|prefer|instead of|rather than)\b{_NON_TERM}*{_SENT})",
    re.IGNORECASE,
)
# Dead-end phrasing.
RE_DEADEND = re.compile(
    rf"({_NON_TERM}*\b(?:didn'?t work|doesn'?t work|that failed|won'?t work|"
    rf"abandon(?:ed)?|gave up on|tried .* but|that approach (?:failed|broke))\b"
    rf"{_NON_TERM}*{_SENT})",
    re.IGNORECASE,
)
# TODO-ish phrasing.
RE_TODO = re.compile(
    rf"({_NON_TERM}*\b(?:TODO|FIXME|next step|next we|still need to|need to "
    rf"(?:add|fix|implement)|let me know if)\b{_NON_TERM}*{_SENT})",
    re.IGNORECASE,
)
# User preference phrasing.
RE_PREF = re.compile(
    rf"({_NON_TERM}*\b(?:I (?:prefer|want|like|always|never)|don'?t use|"
    rf"please don'?t|make sure to)\b{_NON_TERM}*{_SENT})",
    re.IGNORECASE,
)


@dataclass
class Extract:
    files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    dead_ends: List[str] = field(default_factory=list)
    todos: List[str] = field(default_factory=list)
    preferences: List[str] = field(default_factory=list)
    code_blocks: List[tuple] = field(default_factory=list)  # (lang, body)

    def to_hint_block(self) -> str:
        """Compact bullet summary to feed an LLM as a hint."""
        lines = ["<EXTRACTED_HINTS>"]
        if self.files:
            lines.append("Files mentioned: " + ", ".join(sorted(set(self.files))[:30]))
        for label, items in [
            ("Possible decisions", self.decisions),
            ("Possible dead ends", self.dead_ends),
            ("Possible TODOs", self.todos),
            ("Possible user preferences", self.preferences),
            ("Errors observed", self.errors),
        ]:
            uniq = _dedupe(items)[:15]
            if uniq:
                lines.append(f"\n{label}:")
                lines.extend(f"- {s.strip()}" for s in uniq)
        lines.append("</EXTRACTED_HINTS>")
        return "\n".join(lines)


def _dedupe(items: List[str]) -> List[str]:
    seen, out = set(), []
    for s in items:
        k = re.sub(r"\s+", " ", s.strip().lower())
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return out


def extract(transcript: str) -> Extract:
    e = Extract()
    # Strip code blocks first so prose regex don't hit code.
    code_iter = list(RE_CODE_FENCE.finditer(transcript))
    for m in code_iter:
        e.code_blocks.append((m.group(1), m.group(2)))
    prose = RE_CODE_FENCE.sub(" ", transcript)

    e.files = _dedupe(RE_FILE_PATH.findall(transcript))
    e.errors = _dedupe(RE_ERROR.findall(transcript))
    e.decisions = _dedupe(RE_DECISION.findall(prose))
    e.dead_ends = _dedupe(RE_DEADEND.findall(prose))
    e.todos = _dedupe(RE_TODO.findall(prose))
    e.preferences = _dedupe(RE_PREF.findall(prose))
    return e
