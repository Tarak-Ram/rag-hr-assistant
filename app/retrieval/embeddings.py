"""Embedding providers.

`OpenAIEmbedder` is used in production. `HashingEmbedder` is a deterministic,
dependency-free stand-in so that tests, CI and the offline demo run without an
API key. It is a feature-hashed bag of stemmed words and bigrams: weak
semantically, but stable and good enough to exercise the full pipeline.
"""

import hashlib
import math
import re
from typing import Protocol

from app.config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "the", "is", "are", "of", "to", "in", "for", "on", "and", "or", "do",
    "does", "i", "my", "me", "can", "how", "what", "when", "many", "much", "be",
    "it", "at", "by", "with", "from", "per", "if", "get", "we", "our", "you", "your",
}


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        # Very light stemming so "days"/"day" and "claims"/"claim" match.
        for suffix in ("ing", "es", "s"):
            if len(tok) > 4 and tok.endswith(suffix):
                tok = tok[: -len(suffix)]
                break
        tokens.append(tok)
    return tokens


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 768):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for feat in features:
            h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAIEmbedder:
    def __init__(self, settings: Settings):
        from langchain_openai import OpenAIEmbeddings

        self._client = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.dim = len(self._client.embed_query("dimension probe"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


def build_embedder(settings: Settings) -> Embedder:
    if settings.provider == "openai":
        return OpenAIEmbedder(settings)
    return HashingEmbedder()
