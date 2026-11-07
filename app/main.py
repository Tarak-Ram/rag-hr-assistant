"""FastAPI service."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.assistant import Assistant

log = logging.getLogger("hr-assistant")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    employee_id: str | None = Field(default=None, description="Caller's employee ID, for personal lookups")
    debug: bool = False


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    refused: bool
    route: str
    latency_ms: float
    trace: list[dict] | None = None


assistant: Assistant | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global assistant
    assistant = Assistant()
    n = assistant.ingest()
    log.info("indexed %d chunks (provider=%s)", n, assistant.settings.provider)
    yield


app = FastAPI(title="RAG HR Assistant", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": assistant.indexed if assistant else 0}


@app.post("/ingest")
def ingest() -> dict:
    return {"indexed_chunks": assistant.ingest()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if assistant is None or assistant.indexed == 0:
        raise HTTPException(503, "index not ready")
    result = assistant.ask(req.question, req.employee_id)
    log.info("route=%s refused=%s latency_ms=%s", result.route, result.refused, result.latency_ms)
    return ChatResponse(
        answer=result.answer,
        citations=result.citations,
        refused=result.refused,
        route=result.route,
        latency_ms=result.latency_ms,
        trace=result.trace if req.debug else None,
    )
