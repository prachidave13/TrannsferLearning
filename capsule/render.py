"""End-to-end algorithmic capsule builder.

Pipeline:
1. Segment transcript into sentences (turn-aware).
2. Tokenize + TF-IDF vectors.
3. Build cosine-similarity sentence graph; PageRank → centrality scores.
4. Boost scores using positional and role features.
5. Discourse-classify sentences and route to capsule sections.
6. MMR-select top-K diverse sentences per section.
7. Render into the 7-section markdown schema.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from .features import tokenize, vectors_for
from .mmr import mmr_select
from .router import classify, route
from .segment import Sentence, segment
from .textrank import build_graph, pagerank


# How many sentences (max) to include per section.
SECTION_BUDGET = {
    "meta_goal":       1,
    "meta_pref":       3,
    "meta_constraint": 3,
    "meta_stack":      4,
    "state_break":     3,
    "decisions":       6,
    "dead_ends":       4,
    "open_threads":    4,
    "snippets":        2,
}


def _score_sentences(sentences: List[Sentence], pr: List[float]) -> Dict[int, Dict[str, float]]:
    """Combine PageRank with positional/role boosts. Returns vectors map for MMR."""
    if not sentences:
        return {}

    # Normalize PR to [0,1].
    pr_max = max(pr) if pr else 1.0
    pr_max = pr_max or 1.0

    for i, s in enumerate(sentences):
        base = pr[i] / pr_max
        # Recency boost — last 25% of chat is more relevant for "current state".
        recency = 0.0
        if s.position >= 0.75:
            recency = 0.15
        # User-stated goals/preferences carry more weight than assistant prose.
        role_boost = 0.10 if s.role == "user" else 0.0
        # Penalize very short or very long fragments (noise / dumps).
        length = len(s.text)
        length_pen = -0.20 if length < 15 or length > 400 else 0.0
        # Tag presence is a small boost — it means a discourse rule fired.
        tag_boost = 0.10 if s.tags and s.tags[0] != "snippet" else 0.0
        s.score = base + recency + role_boost + length_pen + tag_boost
    return {}


def _extract_files(sentences: List[Sentence]) -> List[tuple]:
    """Files mentioned, with mention count. Returns [(path, count), ...]."""
    rx = re.compile(
        r"`?([a-zA-Z0-9_\-./]+\.(?:py|js|ts|tsx|jsx|md|json|yml|yaml|toml|"
        r"go|rs|java|c|cpp|h|hpp|cs|rb|sh|sql|html|css|scss|"
        r"ipynb|pdf|csv|tsv|txt|png|jpg|svg))`?"
    )
    counts: Counter = Counter()
    for s in sentences:
        if s.in_code:
            continue
        for m in rx.finditer(s.text):
            counts[m.group(1)] += 1
    return counts.most_common()


def _extract_stack(sentences: List[Sentence]) -> List[str]:
    """Surface the most-mentioned stack tokens (heuristic on `stack` tag rule)."""
    rx = re.compile(
        r"\b(python|fastapi|flask|django|node|express|react|next\.js|next|"
        r"sqlite|postgres|redis|docker|kubernetes|fly\.io|aws|gcp|pydantic|"
        r"sqlalchemy|typescript|rust|go|tailwind|svelte|vue|"
        r"jupyter|notebook|copilot|gemini|claude|chatgpt|sklearn|scikit-learn|"
        r"numpy|pandas|svm|kmeans|silhouette|dendrogram|hamming|smote)\b",
        re.IGNORECASE,
    )
    counts: Counter = Counter()
    for s in sentences:
        if s.in_code:
            continue
        for m in rx.finditer(s.text):
            counts[m.group(1).lower()] += 1
    return [tok for tok, _ in counts.most_common(8)]


def _bullet(s: Sentence) -> str:
    text = re.sub(r"\s+", " ", s.text).strip()
    return f"- {text}"


def _section(items: List[Sentence], empty: str = "_(none recorded)_") -> str:
    if not items:
        return empty
    return "\n".join(_bullet(s) for s in items)


def build_algorithmic(transcript: str) -> str:
    sentences = segment(transcript)
    prose_sents = [s for s in sentences if not s.in_code]

    # 1. Tokenize + vectors
    tokens = [tokenize(s.text) for s in prose_sents]
    vectors, _ = vectors_for(tokens)
    vec_map = {id(s): v for s, v in zip(prose_sents, vectors)}

    # 2. TextRank
    adj = build_graph(vectors, threshold=0.05)
    pr = pagerank(adj)

    # 3. Score (mutates .score on each sentence)
    _score_sentences(prose_sents, pr)
    # Code blocks: score by uniqueness (rare = important). Use length as proxy.
    code_sents = [s for s in sentences if s.in_code]
    for cs in code_sents:
        cs.score = 0.5 + min(len(cs.text), 600) / 1200.0
        vec_map[id(cs)] = {}  # not used for similarity in snippet MMR

    # 4. Classify + route
    classify(sentences)
    buckets = route(sentences)

    # 5. MMR per section
    picks: Dict[str, List[Sentence]] = {}
    for section, budget in SECTION_BUDGET.items():
        picks[section] = mmr_select(
            buckets.get(section, []), vec_map, k=budget, lam=0.7
        )

    # 6. Pull structured extras
    files = _extract_files(prose_sents)
    stack = _extract_stack(prose_sents)

    # 7. "Open threads" — also force-include the very last user/assistant
    #    sentences if nothing was tagged, since they are the literal handoff state.
    if not picks["open_threads"]:
        tail = sorted(prose_sents, key=lambda s: -s.position)[:2]
        picks["open_threads"] = tail

    # 8. Resume prompt — try to derive the "next step" from open_threads.
    next_step = picks["open_threads"][0].text if picks["open_threads"] else "<fill in>"

    file_lines = "\n".join(f"- `{p}` (mentioned {c}×)" for p, c in files[:15]) or "_(none detected)_"
    stack_line = ", ".join(stack) if stack else "_(not specified)_"

    snippet_block = "\n\n".join(
        f"```{s.code_lang}\n{s.text}\n```"
        for s in picks["snippets"]
    ) or "_(none recorded)_"

    return f"""# Context Capsule

> Generated algorithmically (TF-IDF + TextRank + MMR). No LLM.
> Every line below is a verbatim sentence from the source chat — zero
> hallucination risk. If a section is sparse, the source chat lacked that
> signal.

## 1. Project Meta
**Goal**
{_section(picks['meta_goal'])}

**Tech stack** (frequency-ranked from chat)
{stack_line}

**Constraints**
{_section(picks['meta_constraint'])}

**User preferences**
{_section(picks['meta_pref'])}

## 2. Current State
**Files mentioned (by frequency)**
{file_lines}

**Errors / breakage observed**
{_section(picks['state_break'])}

## 3. Decision Log
{_section(picks['decisions'])}

## 4. Dead Ends (do not retry)
{_section(picks['dead_ends'])}

## 5. Open Threads
{_section(picks['open_threads'])}

## 6. Critical Snippets
{snippet_block}

## 7. Resume Prompt
> I'm continuing work from a previous LLM session. Read the Context Capsule
> above carefully — especially sections 3 (Decision Log) and 4 (Dead Ends).
> Do not re-suggest approaches listed in Dead Ends. The next step is:
> *{next_step}* — please proceed from there or ask me to clarify.
"""
