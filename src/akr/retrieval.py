"""BM25 keyword retrieval, implemented from scratch (no external services or model downloads)."""

from __future__ import annotations

import math
import re
from collections import Counter

from akr.chunking import Chunk

STOPWORDS = set(
    "a an the and or of to in on for is are be by with at as it its this that these those from can may must if "
    "when what which who whom how do does did i my our we you your they their there than then so not no any "
    "all each per into about after before up out will would should could has have had was were been being".split()
)


def stem(token: str) -> str:
    """Very light suffix stripping so that "fees" matches "fee" and "cancelled" matches "cancel"."""
    for suffix in ("ations", "ation", "ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    if token.endswith("ll"):  # "cancell(ed)" -> "cancel"
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def expand_query(query: str, glossary: dict[str, str]) -> str:
    """Append document vocabulary for everyday words or phrases found in the query."""
    lowered = query.lower()
    extra = [target for phrase, target in glossary.items() if re.search(rf"\b{re.escape(phrase)}\b", lowered)]
    return f"{query} {' '.join(extra)}" if extra else query


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        # Title and section heading are repeated so headings weigh more than body text.
        self.docs = [tokenize(f"{c.title} {c.section} {c.section} {c.text}") for c in chunks]
        self.tfs = [Counter(d) for d in self.docs]
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.docs), 1)
        df = Counter(term for d in self.docs for term in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query_terms: list[str], i: int) -> float:
        tf, length, s = self.tfs[i], self.lengths[i], 0.0
        for t in query_terms:
            if t in tf:
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg_len))
        return s

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        scored = [(self.chunks[i], self.score(terms, i)) for i in range(len(self.chunks))]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
