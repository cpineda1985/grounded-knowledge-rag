from akr.chunking import chunk_markdown, load_corpus, slugify

from conftest import ROOT


def test_one_chunk_per_section_with_citable_ids():
    text = "# Payments\n\n> disclaimer\n\n## Deposit\n\nPay 30%.\n\n## Late payments\n\nCall the client."
    chunks = chunk_markdown("payments", text)
    assert [c.id for c in chunks] == ["payments#deposit", "payments#late-payments"]
    assert chunks[0].title == "Payments"
    assert "disclaimer" not in chunks[0].text


def test_document_without_sections_becomes_overview():
    chunks = chunk_markdown("faq", "# FAQ\n\nJust one paragraph.")
    assert [c.id for c in chunks] == ["faq#overview"]


def test_slugify():
    assert slugify("Office Hours, Branches and Roles") == "office-hours-branches-and-roles"


def test_sample_corpus_loads_with_unique_ids():
    chunks = load_corpus(ROOT / "data" / "sample_agency")
    assert len(chunks) >= 40
    assert len({c.id for c in chunks}) == len(chunks)
