def test_keyword_query_finds_exact_term(assistant):
    hits = assistant.retriever.search("POSH training", k=3)
    assert any("First 30 Days" in h.chunk.section for h in hits)


def test_results_are_ranked_by_fused_score(assistant):
    hits = assistant.retriever.search("hotel limit per night", k=4)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0].chunk.section == "Hotels"


def test_fusion_uses_both_retrievers(assistant):
    hits = assistant.retriever.search("internet reimbursement remote work", k=4)
    top = hits[0]
    assert top.dense_rank is not None and top.sparse_rank is not None
