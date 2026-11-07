"""The LangGraph workflow.

    route ──► smalltalk ─────────────────────────────────► END
      │
      ├──► leave_balance (tool) ─────────────────────────► END
      │
      └──► retrieve ──► grade ──► generate ──► verify ───► END
              ▲           │                      │
              │           ▼ (no evidence)        ▼ (no valid citation)
              └──── rewrite (max 1)  ──►  refuse ◄┘

Design choices:
- Grading before generation: the model never sees passages that failed the
  relevance check, which cuts hallucinations from loosely related context.
- One query rewrite: recovers colloquial questions ("vacation", "wfh") without
  letting the graph loop indefinitely.
- Verify step: an answer with no valid citation is replaced by a refusal.
  "I don't know" is a correct answer; a confident wrong one is not.
"""

import re
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.config import Settings
from app.llm import LLM, Route
from app.retrieval.chunking import Chunk
from app.retrieval.hybrid import HybridRetriever
from app.tools import get_leave_balance

REFUSAL = "I don't know based on the available policies. Please contact your HR business partner."
MAX_REWRITES = 1


class AgentState(TypedDict, total=False):
    question: str
    employee_id: str | None
    route: Route
    query: str
    rewrites: int
    candidates: list[Chunk]
    evidence: list[Chunk]
    answer: str
    citations: list[str]
    refused: bool
    trace: list[dict]


def _log(state: AgentState, step: str, **info) -> list[dict]:
    return [*state.get("trace", []), {"step": step, "t": round(time.perf_counter(), 4), **info}]


def build_graph(llm: LLM, retriever: HybridRetriever, settings: Settings):
    def route(state: AgentState) -> AgentState:
        r = llm.route(state["question"])
        return {"route": r, "query": state["question"], "rewrites": 0, "trace": _log(state, "route", route=r)}

    def smalltalk(state: AgentState) -> AgentState:
        return {
            "answer": "Hi! Ask me about leave, remote work, travel expenses, onboarding or the code of conduct.",
            "citations": [],
            "refused": False,
            "trace": _log(state, "smalltalk"),
        }

    def leave_balance(state: AgentState) -> AgentState:
        result = get_leave_balance(state.get("employee_id"))
        if not result.ok:
            msg = (
                "I need your employee ID to look up your leave balance."
                if result.error == "missing_employee_id"
                else "I couldn't find that employee ID in the HR system."
            )
            return {"answer": msg, "citations": [], "refused": False, "trace": _log(state, "tool", error=result.error)}
        d = result.data
        answer = (
            f"{d['name']}, you have {d['annual_leave_left']} annual leave days, "
            f"{d['sick_leave_left']} sick leave days and {d['floating_holidays_left']} floating holidays left."
        )
        return {"answer": answer, "citations": ["HRIS: leave balance"], "refused": False,
                "trace": _log(state, "tool", tool="get_leave_balance")}

    def retrieve(state: AgentState) -> AgentState:
        hits = retriever.search(state["query"], k=settings.top_k)
        return {"candidates": [h.chunk for h in hits],
                "trace": _log(state, "retrieve", query=state["query"], hits=[h.chunk.citation for h in hits])}

    def grade(state: AgentState) -> AgentState:
        # Grade against the (possibly rewritten) query so a rewrite can rescue colloquial questions.
        kept = [c for c in state["candidates"] if llm.grade(state["query"], c) >= settings.min_relevance]
        return {"evidence": kept, "trace": _log(state, "grade", kept=len(kept), of=len(state["candidates"]))}

    def rewrite(state: AgentState) -> AgentState:
        q = llm.rewrite(state["question"])
        return {"query": q, "rewrites": state.get("rewrites", 0) + 1, "trace": _log(state, "rewrite", query=q)}

    def generate(state: AgentState) -> AgentState:
        return {"answer": llm.generate(state["question"], state["evidence"]), "trace": _log(state, "generate")}

    def verify(state: AgentState) -> AgentState:
        """Guardrail: keep only citations that point at real evidence."""
        evidence = state["evidence"]
        answer = state["answer"]
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        valid = sorted(n for n in cited if 1 <= n <= len(evidence))
        if not valid or answer.lower().startswith("i don't know"):
            return {"answer": REFUSAL, "citations": [], "refused": True, "trace": _log(state, "verify", ok=False)}
        # Remove any hallucinated citation numbers from the text.
        for n in cited - set(valid):
            answer = answer.replace(f"[{n}]", "")
        return {
            "answer": answer.strip(),
            "citations": [evidence[n - 1].citation for n in valid],
            "refused": False,
            "trace": _log(state, "verify", ok=True),
        }

    def refuse(state: AgentState) -> AgentState:
        return {"answer": REFUSAL, "citations": [], "refused": True, "trace": _log(state, "refuse")}

    def after_grade(state: AgentState) -> str:
        if state["evidence"]:
            return "generate"
        return "rewrite" if state.get("rewrites", 0) < MAX_REWRITES else "refuse"

    g = StateGraph(AgentState)
    for name, fn in [("route", route), ("smalltalk", smalltalk), ("leave_balance", leave_balance),
                     ("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
                     ("generate", generate), ("verify", verify), ("refuse", refuse)]:
        g.add_node(name, fn)

    g.set_entry_point("route")
    g.add_conditional_edges("route", lambda s: s["route"],
                            {"policy": "retrieve", "leave_balance": "leave_balance", "smalltalk": "smalltalk"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    for terminal in ("smalltalk", "leave_balance", "verify", "refuse"):
        g.add_edge(terminal, END)
    return g.compile()
