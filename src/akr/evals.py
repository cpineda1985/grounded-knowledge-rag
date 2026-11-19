"""Evaluation: retrieval quality (offline, free) and answer quality (calls the model)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from akr.answer import KnowledgeBase


def load_questions(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def retrieval_eval(kb: KnowledgeBase, questions: list[dict]) -> dict:
    """Hit@1, Hit@k and MRR on answerable questions; gate behaviour on all questions."""
    answerable = [q for q in questions if q["expected"]]
    unanswerable = [q for q in questions if not q["expected"]]
    rows, hit1, hitk, rr = [], 0, 0, 0.0
    for q in answerable:
        hits = kb.retrieve(q["question"])
        ids = [c.id for c, _ in hits]
        rank = next((i + 1 for i, cid in enumerate(ids) if cid in q["expected"]), None)
        hit1 += rank == 1
        hitk += rank is not None
        rr += 1 / rank if rank else 0.0
        rows.append({"id": q["id"], "rank": rank, "top": ids[:1], "top_score": round(hits[0][1], 2) if hits else 0.0})
    gate_rows = []
    for q in questions:
        hits = kb.retrieve(q["question"])
        top = hits[0][1] if hits else 0.0
        gate_rows.append({"id": q["id"], "answerable": bool(q["expected"]), "top_score": round(top, 2),
                          "gated": top < kb.min_score})
    n = max(len(answerable), 1)
    return {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "top_k": kb.top_k,
        "hit_at_1": hit1 / n,
        "hit_at_k": hitk / n,
        "mrr": rr / n,
        "gate_min_score": kb.min_score,
        "gate_blocked_answerable": sum(r["gated"] for r in gate_rows if r["answerable"]),
        "gate_blocked_unanswerable": sum(r["gated"] for r in gate_rows if not r["answerable"]),
        "rows": rows,
        "gate_rows": gate_rows,
    }


def generation_eval(kb: KnowledgeBase, questions: list[dict]) -> dict:
    """Runs every question through the model and checks grounding with deterministic rules."""
    rows = []
    for q in questions:
        a = kb.ask(q["question"])
        answerable = bool(q["expected"])
        rows.append({
            **{k: v for k, v in asdict(a).items() if k != "retrieved"},
            "id": q["id"],
            "answerable": answerable,
            "cited_expected": any(c in q["expected"] for c in a.citations) if answerable else None,
            "correct": (not a.refused and not a.invalid_citations and any(c in q["expected"] for c in a.citations))
            if answerable else a.refused,
        })
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    return {
        "model": next((r["model"] for r in rows if r["model"]), None),
        "questions": len(rows),
        "answered_with_expected_citation": sum(bool(r["cited_expected"]) for r in ans) / max(len(ans), 1),
        "answers_with_invalid_citations": sum(bool(r["invalid_citations"]) for r in rows),
        "wrong_refusals_on_answerable": sum(r["refused"] for r in ans),
        "correct_refusals_on_unanswerable": sum(r["refused"] for r in una) / max(len(una), 1),
        "refused_by_gate_for_free": sum(r["gated"] for r in rows),
        "overall_correct": sum(r["correct"] for r in rows) / max(len(rows), 1),
        "total_cost_usd": round(sum(costs), 6),
        "avg_cost_usd_per_question": round(sum(costs) / max(len(costs), 1), 6),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / max(len(rows), 1), 2),
        "total_input_tokens": sum(r["input_tokens"] for r in rows),
        "total_output_tokens": sum(r["output_tokens"] for r in rows),
        "rows": rows,
    }
