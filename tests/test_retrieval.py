from akr.retrieval import expand_query, stem, tokenize


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The fees for cancelled bookings") == ["fee", "cancel", "booking"]
    assert stem("cancellations") == stem("cancel") == stem("cancelled") == "cancel"


def test_glossary_expansion_matches_whole_words_and_phrases():
    glossary = {"bag": "baggage", "money back": "refund"}
    assert expand_query("Lost bag", glossary) == "Lost bag baggage"
    assert expand_query("When do I get my money back?", glossary).endswith("refund")
    assert expand_query("A handbag question", glossary) == "A handbag question"


def test_deposit_question_finds_deposit_section(make_kb):
    kb, _ = make_kb()
    top_chunk, _ = kb.retrieve("How much deposit is needed to confirm a vacation package?")[0]
    assert top_chunk.id == "payments#deposit"


def test_glossary_bridges_everyday_words(make_kb):
    kb, _ = make_kb()
    ids = [c.id for c, _ in kb.retrieve("The airline lost my client's suitcase")]
    assert "baggage#lost-or-delayed-baggage" in ids
