"""Evaluation harness.

Metrics
- route_accuracy      : the graph picked the expected branch
- retrieval_hit@k     : the expected policy section was among the top-k retrieved chunks
- answer_correctness  : the answer contains the expected key fact(s)
- citation_accuracy   : the expected section is among the answer's citations
- refusal_precision / refusal_recall : refusing exactly when the answer is not in the policies
- faithfulness (LLM-as-judge, only with HRA_PROVIDER=openai)
- latency p50 / p95

Usage:
    python -m evals.run_evals                 # print report, write evals/results.json
    python -m evals.run_evals --gate          # exit non-zero if below thresholds (used in CI)
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from app.assistant import Assistant

HERE = Path(__file__).parent

# CI fails if a change drops any metric below these.
THRESHOLDS = {
    "route_accuracy": 0.95,
    "retrieval_hit@k": 0.85,
    "answer_correctness": 0.75,
    "refusal_recall": 1.0,
    "refusal_precision": 0.9,
}

JUDGE_PROMPT = """You are grading an HR assistant. Given the passages it was shown and its answer,
reply with only a number from 0 to 1: the fraction of claims in the answer that are supported by the passages.

Passages:
{context}

Answer:
{answer}"""


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def judge_faithfulness(assistant: Assistant, question: str, answer: str) -> float | None:
    if assistant.settings.provider != "openai":
        return None
    hits = assistant.retriever.search(question, k=assistant.settings.top_k)
    context = "\n\n".join(h.chunk.text for h in hits)
    raw = assistant.llm.chat.invoke(JUDGE_PROMPT.format(context=context, answer=answer)).content
    try:
        return max(0.0, min(1.0, float(raw.strip())))
    except ValueError:
        return None


def ratio(xs: list[bool]) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 1.0


def run(dataset: list[dict]) -> tuple[dict, list[dict]]:
    assistant = Assistant()
    assistant.ingest()

    rows = []
    for case in dataset:
        r = assistant.ask(case["question"], case.get("employee_id"))
        answer_ok = all(s.lower() in r.answer.lower() for s in case.get("must_contain", []))
        expected = case.get("expected_section")
        row = {
            "id": case["id"],
            "route_ok": r.route == case["route"],
            "hit": (expected in r.retrieved) if expected else None,
            "answer_ok": answer_ok if case.get("must_contain") else None,
            "cited_ok": (expected in r.citations) if expected else None,
            "should_refuse": bool(case.get("should_refuse")),
            "refused": r.refused,
            "latency_ms": r.latency_ms,
            "answer": r.answer,
        }
        if expected and not r.refused:
            row["faithfulness"] = judge_faithfulness(assistant, case["question"], r.answer)
        rows.append(row)

    refused = [x for x in rows if x["refused"]]
    should = [x for x in rows if x["should_refuse"]]
    latencies = sorted(x["latency_ms"] for x in rows)
    faith = [x["faithfulness"] for x in rows if x.get("faithfulness") is not None]

    metrics = {
        "cases": len(rows),
        "provider": assistant.settings.provider,
        "route_accuracy": ratio([x["route_ok"] for x in rows]),
        "retrieval_hit@k": ratio([x["hit"] for x in rows if x["hit"] is not None]),
        "answer_correctness": ratio([x["answer_ok"] for x in rows if x["answer_ok"] is not None]),
        "citation_accuracy": ratio([x["cited_ok"] for x in rows if x["cited_ok"] is not None]),
        "refusal_precision": ratio([x["should_refuse"] for x in refused]),
        "refusal_recall": ratio([x["refused"] for x in should]),
        "faithfulness": round(statistics.mean(faith), 3) if faith else None,
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
    }
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=HERE / "dataset.jsonl")
    parser.add_argument("--gate", action="store_true", help="fail if metrics fall below thresholds")
    args = parser.parse_args()

    metrics, rows = run(load_dataset(args.dataset))
    (HERE / "results.json").write_text(json.dumps({"metrics": metrics, "cases": rows}, indent=2))

    print(f"\n{'metric':<22}{'value':>10}{'threshold':>12}")
    print("-" * 44)
    failed = []
    for k, v in metrics.items():
        if k in ("cases", "provider"):
            continue
        t = THRESHOLDS.get(k)
        flag = ""
        if t is not None and v is not None and v < t:
            failed.append(k)
            flag = "  FAIL"
        print(f"{k:<22}{str(v):>10}{(str(t) if t is not None else '-'):>12}{flag}")

    misses = [r for r in rows if r["answer_ok"] is False or r["route_ok"] is False
              or (r["should_refuse"] != r["refused"])]
    if misses:
        print("\nFailing cases:")
        for r in misses:
            print(f"  {r['id']}: {r['answer'][:110]}")

    if args.gate and failed:
        print(f"\nEval gate failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
