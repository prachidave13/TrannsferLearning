"""Discourse classifier + section router.

Tags each sentence with zero or more discourse roles, then routes them to
the appropriate capsule section. Rule-based — no model. Each rule is a
weighted regex; tags accumulate, the highest-weighted tag wins as primary.
"""
from __future__ import annotations

import re
from typing import Dict, List

from .segment import Sentence

# (tag, regex, weight). Weight breaks ties when multiple rules fire.
_RULES = [
    ("goal", re.compile(r"\b(i (?:want|need)|let'?s build|goal is|trying to|build (?:a|an)|we'?re building|tell me|help me|fix (?:it|this|the)|if (?:i'?m|im) missing)\b", re.I), 3.0),
    ("preference", re.compile(r"\b(i (?:prefer|like|always|never)|don'?t use|please don'?t|make sure (?:to|not)|i hate|stop\b|do not)\b", re.I), 3.0),
    # Decisions: collaborative ("let's use X") OR prescriptive ("change X to Y", "needs to be").
    ("decision", re.compile(r"\b(let'?s use|we'?ll (?:use|go with)|decided to|chose|going with|switching to|instead of|rather than|we should use|change\s+.*\s+to|needs? to be|should be (?:changed|set|fixed)|fix\s*\d+|fix\s*[—-]|the fix is)\b", re.I), 2.5),
    # Dead ends: failed attempts OR retracted/incorrect prior claims.
    ("dead_end", re.compile(r"\b(didn'?t work|doesn'?t work|that failed|won'?t work|abandon(?:ed)?|gave up|tried .* but|approach failed|that broke|too slow|round trips were slow|was wrong|i apologize|incorrect|bad first review|contradicting|truncated reading|i was flagging|do not add|stop\b)\b", re.I), 3.5),
    # Errors / issues / problems found during review.
    ("error", re.compile(r"\b(error|exception|traceback|operationalerror|failed|undefined|not found|cannot find|panic:|is (?:cut off|truncated|incomplete|wrong|missing|identical)|are identical|missing\b|hardcoded|issues? to fix|not (?:fixed|implemented))\b", re.I), 2.0),
    # TODO / next-step / status-of-work.
    ("todo", re.compile(r"(\bnext\s*:|\b(todo|fixme|next step|next we|still need|need to (?:add|fix|implement|build)|let'?s (?:tackle|do) .* next|pause here|ran perfectly|run all|good to (?:run|go)|complete and correct|both fixes|implemented|all .* executed|no errors|you'?re good)\b)", re.I), 3.0),
    ("constraint", re.compile(r"\b(must|should be deployable|deploy(?:ed)? on|target is|deadline|performance|under .* load|the (?:homework|assignment|prompt) (?:asks|specifies|says))\b", re.I), 1.5),
    # File mentions: extended to notebooks, docs, images, csv, etc.
    ("file_mention", re.compile(
        r"`[^`]+\.(py|js|ts|tsx|jsx|md|json|yml|yaml|toml|go|rs|sql|sh|ipynb|pdf|csv|tsv|txt|png|jpg|svg|html|css|scss)`"
        r"|\b[\w/\-]+\.(py|js|ts|tsx|jsx|md|json|yml|yaml|toml|go|rs|sql|sh|ipynb|pdf|csv|tsv|txt|png|jpg|svg|html|css|scss)\b"
    ), 1.0),
    ("schema", re.compile(r"\b(create table|schema|interface|type \w+ =|class \w+|def \w+\(|function \w+\()", re.I), 1.5),
    ("stack", re.compile(r"\b(python|fastapi|flask|django|node|express|react|next|sqlite|postgres|redis|docker|kubernetes|fly\.io|aws|gcp|pydantic|sqlalchemy|nanoid|typescript|rust|go|jupyter|notebook|copilot|gemini|claude|chatgpt|sklearn|scikit-learn|numpy|pandas|svm|kmeans|silhouette|dendrogram|hamming)\b", re.I), 1.0),
]


def classify(sentences: List[Sentence]) -> None:
    """Mutates `sentences` in place, populating `.tags`."""
    for s in sentences:
        if s.in_code:
            s.tags = ["snippet"]
            continue
        hits: List[tuple] = []
        for tag, rx, w in _RULES:
            if rx.search(s.text):
                hits.append((tag, w))
        # Sort by weight desc; primary tag first.
        hits.sort(key=lambda x: -x[1])
        s.tags = [t for t, _ in hits]


# How tags map into capsule sections. A sentence may appear in multiple
# sections (e.g., a goal sentence is also project meta).
SECTION_RULES = {
    "meta_goal":        {"tags": {"goal"},                  "min_weight": 1},
    "meta_pref":        {"tags": {"preference"},            "min_weight": 1},
    "meta_constraint":  {"tags": {"constraint"},            "min_weight": 1},
    "meta_stack":       {"tags": {"stack"},                 "min_weight": 1},
    "state_files":      {"tags": {"file_mention"},          "min_weight": 1},
    "state_break":      {"tags": {"error"},                 "min_weight": 1},
    "decisions":        {"tags": {"decision"},              "min_weight": 1},
    "dead_ends":        {"tags": {"dead_end"},              "min_weight": 1},
    "open_threads":     {"tags": {"todo"},                  "min_weight": 1},
    "snippets":         {"tags": {"snippet"},               "min_weight": 1},
}


def route(sentences: List[Sentence]) -> Dict[str, List[Sentence]]:
    buckets: Dict[str, List[Sentence]] = {k: [] for k in SECTION_RULES}
    for s in sentences:
        for section, rule in SECTION_RULES.items():
            if any(t in rule["tags"] for t in s.tags):
                buckets[section].append(s)
    return buckets
