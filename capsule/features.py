"""Tokenization, TF-IDF vectors, cosine similarity. Pure stdlib.

TF-IDF formula:
    tf(t, d)   = count(t in d) / max(1, sum(counts in d))
    idf(t)     = log( (1 + N) / (1 + df(t)) ) + 1
    tfidf(t,d) = tf(t,d) * idf(t)

Cosine similarity:
    cos(u, v) = (u . v) / (||u|| * ||v||)
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

# Minimal English stopword list — enough to clean TF-IDF without a dep.
STOPWORDS = set("""
a an the and or but if then else for to of in on at by with from as is are was
were be been being do does did have has had not no yes it its this that these
those i you he she we they me him her us them my your his their our so than
about into out up down over under again very can will would could should may
might just also more most other some such only own same too any all each few
which who whom whose what when where why how here there now any some
""".split())

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_+\-\.]*[A-Za-z0-9]|[A-Za-z]")


def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in _TOKEN.findall(text)]
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


def compute_idf(docs: List[List[str]]) -> Dict[str, float]:
    n = len(docs)
    df: Counter = Counter()
    for doc in docs:
        for t in set(doc):
            df[t] += 1
    return {t: math.log((1 + n) / (1 + df_t)) + 1.0 for t, df_t in df.items()}


def tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = sum(counts.values())
    return {t: (c / total) * idf.get(t, 0.0) for t, c in counts.items()}


def cosine(u: Dict[str, float], v: Dict[str, float]) -> float:
    if not u or not v:
        return 0.0
    # Iterate over the shorter dict.
    if len(u) > len(v):
        u, v = v, u
    dot = sum(w * v.get(k, 0.0) for k, w in u.items())
    nu = math.sqrt(sum(w * w for w in u.values()))
    nv = math.sqrt(sum(w * w for w in v.values()))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


def vectors_for(sent_tokens: List[List[str]]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    idf = compute_idf(sent_tokens)
    return [tfidf_vector(toks, idf) for toks in sent_tokens], idf
