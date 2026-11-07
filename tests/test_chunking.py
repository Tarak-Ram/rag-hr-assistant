from app.config import get_settings
from app.retrieval.chunking import _split_long, load_corpus


def test_chunks_keep_section_citations():
    chunks = load_corpus(get_settings().docs_dir)
    assert chunks
    assert all(c.doc and c.section and c.text for c in chunks)
    assert "Leave Policy § Sick Leave" in {c.citation for c in chunks}


def test_chunk_ids_are_stable():
    a = [c.id for c in load_corpus(get_settings().docs_dir)]
    b = [c.id for c in load_corpus(get_settings().docs_dir)]
    assert a == b and len(set(a)) == len(a)


def test_long_sections_are_split_within_size():
    text = "\n".join(f"Sentence number {i} about policy details." for i in range(60))
    pieces = _split_long(text, size=300, overlap=50)
    assert len(pieces) > 1
    assert all(len(p) <= 300 + 60 for p in pieces)
