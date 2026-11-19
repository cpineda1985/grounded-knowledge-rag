import pytest

from akr.answer import REFUSAL, extract_citations


def test_valid_citation_is_accepted_and_cost_is_measured(make_kb):
    kb, llm = make_kb("Packages need a 30% deposit within 48 hours [payments#deposit].")
    a = kb.ask("How much deposit is needed to confirm a vacation package?")
    assert a.citations == ["payments#deposit"]
    assert a.invalid_citations == []
    assert not a.refused and not a.gated
    # 1000 input tokens at $0.10/MTok + 100 output tokens at $0.50/MTok
    assert a.cost_usd == pytest.approx(0.00015)
    assert llm.calls == 1


def test_citation_to_a_source_that_was_not_retrieved_is_flagged(make_kb):
    kb, _ = make_kb("Deposit is 30% [payments#deposit]. Also see [made-up#section].")
    a = kb.ask("How much deposit is needed to confirm a vacation package?")
    assert a.invalid_citations == ["made-up#section"]


def test_model_refusal_is_detected(make_kb):
    kb, _ = make_kb(REFUSAL)
    a = kb.ask("How many vacation days do employees get per year?")
    assert a.refused


def test_weak_retrieval_is_refused_without_calling_the_model(make_kb):
    kb, llm = make_kb("should not be used")
    a = kb.ask("Can we sell train tickets in Europe?")
    assert a.gated and a.refused
    assert a.cost_usd == 0 and llm.calls == 0


def test_extract_citations_deduplicates_in_order():
    assert extract_citations("a [x#y] b [z#w] c [x#y]") == ["x#y", "z#w"]
