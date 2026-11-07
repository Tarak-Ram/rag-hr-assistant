"""Hybrid retrieval: dense vectors in Qdrant + BM25 keyword search, fused with RRF.

Why hybrid: HR questions mix paraphrase ("time off when my baby is born") with
exact terms ("POSH", "INR 1,500"). Dense search handles the first, BM25 the
second. Reciprocal Rank Fusion combines the two rankings without having to
calibrate their scores against each other.
"""

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from app.retrieval.chunking import Chunk
from app.retrieval.embeddings import Embedder, tokenize

RRF_K = 60


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # fused RRF score, used for ranking
    dense_rank: int | None = None
    sparse_rank: int | None = None


class HybridRetriever:
    def __init__(self, embedder: Embedder, location: str = ":memory:", collection: str = "hr_policies"):
        self.embedder = embedder
        self.collection = collection
        self.client = QdrantClient(location=location) if location == ":memory:" else QdrantClient(url=location)
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[Chunk]) -> int:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
        )
        vectors = self.embedder.embed_documents([f"{c.section}. {c.text}" for c in chunks])
        self.client.upsert(
            self.collection,
            points=[
                PointStruct(id=i, vector=v, payload={"chunk_id": c.id})
                for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )
        self._chunks = chunks
        self._bm25 = BM25Okapi([tokenize(f"{c.doc} {c.section} {c.text}") for c in chunks])
        return len(chunks)

    def _dense(self, query: str, limit: int) -> list[int]:
        hits = self.client.query_points(
            self.collection, query=self.embedder.embed_query(query), limit=limit
        ).points
        return [int(h.id) for h in hits]

    def _sparse(self, query: str, limit: int) -> list[int]:
        assert self._bm25 is not None, "call index() first"
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked[:limit] if scores[i] > 0]

    def search(self, query: str, k: int = 4, candidates: int = 20) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        dense = self._dense(query, candidates)
        sparse = self._sparse(query, candidates)
        fused: dict[int, ScoredChunk] = {}
        for rank, idx in enumerate(dense):
            sc = fused.setdefault(idx, ScoredChunk(self._chunks[idx], 0.0))
            sc.score += 1 / (RRF_K + rank + 1)
            sc.dense_rank = rank + 1
        for rank, idx in enumerate(sparse):
            sc = fused.setdefault(idx, ScoredChunk(self._chunks[idx], 0.0))
            sc.score += 1 / (RRF_K + rank + 1)
            sc.sparse_rank = rank + 1
        return sorted(fused.values(), key=lambda s: s.score, reverse=True)[:k]
