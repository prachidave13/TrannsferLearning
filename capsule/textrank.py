"""TextRank — PageRank over a sentence-similarity graph (power iteration).

Algorithm (Mihalcea & Tarau 2004):
1. Build undirected weighted graph G where nodes = sentences,
   edge weight w(i,j) = cosine(vec_i, vec_j) if above threshold, else 0.
2. Run weighted PageRank to convergence:
       PR(v) = (1 - d) / N  +  d * Σ_{u → v}  PR(u) * w(u,v) / W(u)
   where W(u) = Σ_x w(u, x) and d (damping) ≈ 0.85.
3. Score for each sentence = PR value.

Pure stdlib, O(N^2) similarity build, O(I·E) iteration.
"""
from __future__ import annotations

from typing import Dict, List

from .features import cosine


def build_graph(vectors: List[Dict[str, float]], threshold: float = 0.05) -> List[Dict[int, float]]:
    n = len(vectors)
    adj: List[Dict[int, float]] = [dict() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            w = cosine(vectors[i], vectors[j])
            if w >= threshold:
                adj[i][j] = w
                adj[j][i] = w
    return adj


def pagerank(
    adj: List[Dict[int, float]],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> List[float]:
    n = len(adj)
    if n == 0:
        return []
    pr = [1.0 / n] * n
    out_w = [sum(neighbors.values()) for neighbors in adj]
    base = (1.0 - damping) / n

    for _ in range(max_iter):
        new = [base] * n
        for u, neighbors in enumerate(adj):
            if out_w[u] == 0:
                # Dangling: distribute evenly.
                share = damping * pr[u] / n
                for v in range(n):
                    new[v] += share
                continue
            contribution = damping * pr[u] / out_w[u]
            for v, w in neighbors.items():
                new[v] += contribution * w
        delta = sum(abs(new[i] - pr[i]) for i in range(n))
        pr = new
        if delta < tol:
            break
    return pr
