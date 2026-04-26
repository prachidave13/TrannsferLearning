"""Maximal Marginal Relevance — pick top-K diverse, high-relevance items.

Carbonell & Goldstein 1998:
    MMR = argmax_{s in candidates \\ S}  [ λ · rel(s)  −  (1 − λ) · max_{s' in S} sim(s, s') ]

`rel`  : relevance score (we use the per-sentence TextRank score)
`sim`  : cosine similarity between TF-IDF vectors
`λ`    : trade-off between relevance (1.0) and diversity (0.0). Default 0.7.
"""
from __future__ import annotations

from typing import Dict, List

from .features import cosine
from .segment import Sentence


def mmr_select(
    candidates: List[Sentence],
    vectors: Dict[int, Dict[str, float]],
    *,
    k: int,
    lam: float = 0.7,
) -> List[Sentence]:
    """Pick up to k sentences from `candidates` using MMR.

    `vectors` maps id(sentence) -> TF-IDF vector.
    """
    if k <= 0 or not candidates:
        return []
    pool = list(candidates)
    selected: List[Sentence] = []
    while pool and len(selected) < k:
        best, best_score = None, -float("inf")
        for s in pool:
            sv = vectors.get(id(s), {})
            if selected:
                max_sim = max(cosine(sv, vectors.get(id(t), {})) for t in selected)
            else:
                max_sim = 0.0
            score = lam * s.score - (1.0 - lam) * max_sim
            if score > best_score:
                best, best_score = s, score
        selected.append(best)
        pool.remove(best)
    return selected
