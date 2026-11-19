"""Command line: `akr search`, `akr ask`, `akr eval`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from akr.answer import KnowledgeBase
from akr.evals import generation_eval, load_questions, retrieval_eval


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from a .env file. Variables already set in the environment win."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def cmd_search(args) -> int:
    kb = KnowledgeBase.from_config(args.config, with_llm=False)
    for chunk, score in kb.retrieve(args.question):
        print(f"{score:6.2f}  {chunk.id}")
    return 0


def cmd_ask(args) -> int:
    kb = KnowledgeBase.from_config(args.config)
    a = kb.ask(args.question)
    print(a.text)
    print()
    print(f"citations: {', '.join(a.citations) or '-'}")
    if a.invalid_citations:
        print(f"INVALID citations: {', '.join(a.invalid_citations)}")
    cost = "n/a" if a.cost_usd is None else f"${a.cost_usd:.6f}"
    gate = " (refused by retrieval gate, no model call)" if a.gated else ""
    print(f"tokens: {a.input_tokens} in / {a.output_tokens} out | cost: {cost} | latency: {a.latency_s:.2f}s{gate}")
    return 0


def print_retrieval(r: dict) -> None:
    print("Retrieval (offline, no model calls)")
    print(f"  answerable questions:   {r['answerable']}")
    print(f"  hit@1:                  {r['hit_at_1']:.0%}")
    print(f"  hit@{r['top_k']}:                  {r['hit_at_k']:.0%}")
    print(f"  MRR:                    {r['mrr']:.2f}")
    print(f"  gate (min_score={r['gate_min_score']}): blocks {r['gate_blocked_unanswerable']}/{r['unanswerable']} "
          f"unanswerable, {r['gate_blocked_answerable']}/{r['answerable']} answerable")
    misses = [row for row in r["rows"] if row["rank"] != 1]
    for row in misses:
        print(f"    {row['id']}: expected at rank {row['rank']}, top was {row['top']}")


def print_generation(g: dict) -> None:
    print(f"Generation ({g['model']}, {g['questions']} questions)")
    print(f"  answered with the expected citation:  {g['answered_with_expected_citation']:.0%}")
    print(f"  answers with invalid citations:       {g['answers_with_invalid_citations']}")
    print(f"  wrong refusals on answerable:         {g['wrong_refusals_on_answerable']}")
    print(f"  correct refusals on unanswerable:     {g['correct_refusals_on_unanswerable']:.0%}")
    print(f"  refused by the gate for $0:           {g['refused_by_gate_for_free']}")
    print(f"  overall correct:                      {g['overall_correct']:.0%}")
    print(f"  total cost: ${g['total_cost_usd']:.4f} | avg per question: ${g['avg_cost_usd_per_question']:.6f} "
          f"| avg latency: {g['avg_latency_s']}s")


def cmd_eval(args) -> int:
    questions = load_questions(args.questions)
    kb = KnowledgeBase.from_config(args.config, with_llm=not args.retrieval_only)
    results = {"retrieval": retrieval_eval(kb, questions)}
    print_retrieval(results["retrieval"])
    if not args.retrieval_only:
        print()
        results["generation"] = generation_eval(kb, questions)
        print_generation(results["generation"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull results: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="akr", description="Domain-agnostic knowledge RAG with citations and evals.")
    parser.add_argument("--config", default="config.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="Show the retrieved sections for a question (no model call).")
    p.add_argument("question")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="Answer a question with citations.")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="Run the evaluation set.")
    p.add_argument("--questions", default="evals/questions.jsonl")
    p.add_argument("--retrieval-only", action="store_true", help="Skip model calls (free, no API key needed).")
    p.add_argument("--out", default="evals/results/latest.json")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    load_dotenv(Path(args.config).resolve().parent / ".env")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
