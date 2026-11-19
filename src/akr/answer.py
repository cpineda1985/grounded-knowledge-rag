"""Grounded answers: retrieve, gate, generate with citations, validate, and measure."""

from __future__ import annotations

import re
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from akr.chunking import Chunk, load_corpus
from akr.llm import AnthropicClient, LLMClient, cost_usd
from akr.retrieval import BM25Index, expand_query

REFUSAL = "I don't know based on the knowledge base."
CITATION_RE = re.compile(r"\[([a-z0-9_-]+#[a-z0-9-]+)\]")

SYSTEM_PROMPT = """You answer questions for {audience} using ONLY the sources provided.

Rules:
- End every sentence that states a fact with the id of its source in square brackets, for example [payments#deposit].
- Use only ids that appear in the sources. Never invent an id.
- If the sources do not contain the answer, reply with exactly: {refusal}
- Do not use outside knowledge, even if you are confident.
- Be concise: at most 5 sentences."""


@dataclass
class Answer:
    question: str
    text: str
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    retrieved: list[tuple[str, float]] = field(default_factory=list)
    refused: bool = False
    gated: bool = False  # refused by the retrieval gate, before any model call
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0
    latency_s: float = 0.0


def format_sources(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(
        f'<source id="{c.id}">\n{c.title} - {c.section}\n{c.text}\n</source>' for c, _ in hits
    )


def extract_citations(text: str) -> list[str]:
    seen: list[str] = []
    for cid in CITATION_RE.findall(text):
        if cid not in seen:
            seen.append(cid)
    return seen


class KnowledgeBase:
    def __init__(
        self,
        chunks: list[Chunk],
        llm: LLMClient | None = None,
        top_k: int = 4,
        min_score: float = 0.0,
        audience: str = "the user",
        pricing: dict | None = None,
        glossary: dict[str, str] | None = None,
    ):
        self.chunks = chunks
        self.index = BM25Index(chunks)
        self.glossary = glossary or {}
        self.llm, self.top_k, self.min_score = llm, top_k, min_score
        self.system = SYSTEM_PROMPT.format(audience=audience, refusal=REFUSAL)
        self.pricing = pricing or {}

    @classmethod
    def from_config(cls, path: str | Path = "config.toml", llm: LLMClient | None = None, with_llm: bool = True):
        path = Path(path)
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
        corpus = Path(cfg["corpus"]["path"])
        if not corpus.is_absolute():
            corpus = path.parent / corpus
        if llm is None and with_llm:
            llm = AnthropicClient(cfg["llm"]["model"], cfg["llm"].get("max_tokens", 600))
        glossary: dict[str, str] = {}
        if cfg["retrieval"].get("use_glossary", True) and (corpus / "glossary.toml").exists():
            glossary = tomllib.loads((corpus / "glossary.toml").read_text(encoding="utf-8")).get("expand", {})
        return cls(
            load_corpus(corpus),
            llm=llm,
            top_k=cfg["retrieval"]["top_k"],
            min_score=cfg["retrieval"]["min_score"],
            audience=cfg["assistant"]["audience"],
            pricing=cfg.get("pricing", {}),
            glossary=glossary,
        )

    def retrieve(self, question: str) -> list[tuple[Chunk, float]]:
        return self.index.search(expand_query(question, self.glossary), self.top_k)

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        hits = self.retrieve(question)
        retrieved = [(c.id, round(s, 3)) for c, s in hits]

        # Gate: weak retrieval means the answer is probably not in the knowledge base. Refuse for $0.
        if not hits or hits[0][1] < self.min_score:
            return Answer(question, REFUSAL, retrieved=retrieved, refused=True, gated=True,
                          latency_s=time.perf_counter() - start)

        if self.llm is None:
            raise RuntimeError("No LLM client configured. Use retrieval only, or set ANTHROPIC_API_KEY.")

        user = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
        completion = self.llm.complete(self.system, user)
        citations = extract_citations(completion.text)
        valid_ids = {c.id for c, _ in hits}
        return Answer(
            question=question,
            text=completion.text.strip(),
            citations=citations,
            invalid_citations=[c for c in citations if c not in valid_ids],
            retrieved=retrieved,
            refused=REFUSAL.lower().rstrip(".") in completion.text.lower(),
            model=completion.model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cost_usd=cost_usd(completion, self.pricing),
            latency_s=time.perf_counter() - start,
        )
