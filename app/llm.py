"""LLM operations used by the graph, behind one small interface.

Each graph node calls a single, narrow operation (route, grade, rewrite,
generate). Keeping them narrow makes each one testable and lets us swap the
model per step, for example a cheap model for routing and a stronger one for
generation.

`OpenAILLM` calls any OpenAI-compatible endpoint. `OfflineLLM` is a
deterministic, rule-based implementation used by tests, CI and the keyless demo.
"""

import re
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from app.config import Settings
from app.retrieval.chunking import Chunk
from app.retrieval.embeddings import tokenize

Route = Literal["policy", "leave_balance", "smalltalk"]


class Grade(BaseModel):
    relevance: float = Field(ge=0, le=1, description="How useful the passage is for answering, 0 to 1")


class RouteDecision(BaseModel):
    route: Route


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.citation})\n{c.text}" for i, c in enumerate(chunks, start=1))


GENERATE_PROMPT = """You are an HR policy assistant. Answer the question using ONLY the numbered passages.
Rules:
- Cite every factual sentence with the passage number in square brackets, e.g. [1].
- If the passages do not contain the answer, reply exactly: I don't know based on the available policies.
- Be concise: at most 4 sentences. Do not invent numbers.

Passages:
{context}

Question: {question}
Answer:"""

ROUTE_PROMPT = """Classify the HR assistant request.
- leave_balance: the user asks about their OWN remaining or used leave days.
- smalltalk: greetings, thanks, or chit-chat with no question.
- policy: anything else, including questions about rules, entitlements and processes.

Request: {question}"""

GRADE_PROMPT = """Rate from 0 to 1 how useful this passage is for answering the question.
0 = unrelated, 1 = directly answers it.

Question: {question}
Passage: {passage}"""

REWRITE_PROMPT = """Rewrite this employee question as a short search query using the vocabulary
an HR policy document would use. Return only the query.

Question: {question}"""


class LLM(Protocol):
    def route(self, question: str) -> Route: ...
    def grade(self, question: str, chunk: Chunk) -> float: ...
    def rewrite(self, question: str) -> str: ...
    def generate(self, question: str, chunks: list[Chunk]) -> str: ...


class OpenAILLM:
    def __init__(self, settings: Settings):
        from langchain_openai import ChatOpenAI

        self.chat = ChatOpenAI(
            model=settings.chat_model,
            temperature=0,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    def route(self, question: str) -> Route:
        out = self.chat.with_structured_output(RouteDecision).invoke(ROUTE_PROMPT.format(question=question))
        return out.route

    def grade(self, question: str, chunk: Chunk) -> float:
        out = self.chat.with_structured_output(Grade).invoke(
            GRADE_PROMPT.format(question=question, passage=chunk.text)
        )
        return out.relevance

    def rewrite(self, question: str) -> str:
        return self.chat.invoke(REWRITE_PROMPT.format(question=question)).content.strip()

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        prompt = GENERATE_PROMPT.format(context=format_context(chunks), question=question)
        return self.chat.invoke(prompt).content.strip()


# ---------------------------------------------------------------------------
# Offline implementation
# ---------------------------------------------------------------------------

_SMALLTALK = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|evening))\b[\s!.]*$", re.I)
_SELF = re.compile(r"\b(i|my|me)\b", re.I)
_LEAVE = re.compile(r"\b(leave|leaves|days off|pto|holidays)\b", re.I)
_REMAINING = re.compile(r"\b(left|remaining|balance)\b", re.I)

# Colloquial phrasing -> policy vocabulary. Used by the rewrite step.
SYNONYMS = {
    "vacation": "annual leave",
    "holiday": "annual leave public holidays",
    "wfh": "remote work hybrid",
    "work from home": "remote work hybrid",
    "baby": "parental leave maternity paternity",
    "maternity": "parental leave maternity",
    "paternity": "parental leave paternity",
    "resign": "notice period",
    "quit": "notice period",
    "sick": "sick leave medical certificate",
    "reimburse": "expense claims",
    "hotel": "hotels per night",
    "moonlight": "outside employment freelance conflict of interest",
    "side job": "outside employment freelance conflict of interest",
    "abroad": "outside India",
}


class OfflineLLM:
    def route(self, question: str) -> Route:
        if _SMALLTALK.match(question):
            return "smalltalk"
        if _SELF.search(question) and _LEAVE.search(question) and _REMAINING.search(question):
            return "leave_balance"
        return "policy"

    def grade(self, question: str, chunk: Chunk) -> float:
        q = set(tokenize(question))
        if not q:
            return 0.0
        c = set(tokenize(f"{chunk.doc} {chunk.section} {chunk.text}"))
        return len(q & c) / len(q)

    def rewrite(self, question: str) -> str:
        lowered = question.lower()
        extra = [v for k, v in SYNONYMS.items() if k in lowered]
        return f"{question} {' '.join(extra)}".strip()

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        """Extractive answer: the most question-relevant sentences, with citations."""
        q = set(tokenize(question))
        scored = []
        for i, chunk in enumerate(chunks, start=1):
            for sent in re.split(r"(?<=[.!?])\s+", chunk.text):
                overlap = len(q & set(tokenize(sent)))
                if overlap:
                    scored.append((overlap, i, sent.strip()))
        if not scored:
            return "I don't know based on the available policies."
        scored.sort(key=lambda t: t[0], reverse=True)
        best = scored[0][0]
        picked = [s for s in scored if s[0] >= max(1, best - 1)][:2]
        return " ".join(f"{sent} [{i}]" for _, i, sent in picked)


def build_llm(settings: Settings) -> LLM:
    if settings.provider == "openai":
        return OpenAILLM(settings)
    return OfflineLLM()
