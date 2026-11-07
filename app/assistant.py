"""Wires settings, retriever, LLM and graph into one object used by the API and evals."""

import time
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.graph import build_graph
from app.llm import build_llm
from app.retrieval.chunking import load_corpus
from app.retrieval.embeddings import build_embedder
from app.retrieval.hybrid import HybridRetriever


@dataclass
class Answer:
    answer: str
    citations: list[str]
    refused: bool
    route: str
    retrieved: list[str]
    latency_ms: float
    trace: list[dict]


class Assistant:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.retriever = HybridRetriever(
            build_embedder(self.settings), self.settings.qdrant_location, self.settings.collection
        )
        self.llm = build_llm(self.settings)
        self.graph = build_graph(self.llm, self.retriever, self.settings)
        self.indexed = 0

    def ingest(self) -> int:
        chunks = load_corpus(self.settings.docs_dir, self.settings.chunk_size, self.settings.chunk_overlap)
        self.indexed = self.retriever.index(chunks)
        return self.indexed

    def ask(self, question: str, employee_id: str | None = None) -> Answer:
        start = time.perf_counter()
        state = self.graph.invoke({"question": question, "employee_id": employee_id, "trace": []})
        return Answer(
            answer=state["answer"],
            citations=state.get("citations", []),
            refused=state.get("refused", False),
            route=state.get("route", ""),
            retrieved=[c.citation for c in state.get("candidates", [])],
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            trace=state.get("trace", []),
        )
