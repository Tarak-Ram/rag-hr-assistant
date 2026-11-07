from app.graph import REFUSAL


def steps(answer):
    return [t["step"] for t in answer.trace]


def test_answers_with_citation(assistant):
    r = assistant.ask("How many sick days do I get per year?")
    assert "12 days" in r.answer
    assert r.citations == ["Leave Policy § Sick Leave"]
    assert not r.refused


def test_refuses_out_of_scope_question(assistant):
    r = assistant.ask("What is the stock option vesting schedule?")
    assert r.refused and r.answer == REFUSAL and r.citations == []


def test_rewrite_runs_at_most_once(assistant):
    r = assistant.ask("Who won the cricket world cup?")
    assert steps(r).count("rewrite") == 1
    assert steps(r)[-1] == "refuse"


def test_colloquial_question_recovered_by_rewrite(assistant):
    r = assistant.ask("Can I go on vacation next week?")
    assert "rewrite" in steps(r)
    assert not r.refused


def test_leave_balance_uses_tool(assistant):
    r = assistant.ask("How many leaves do I have left?", employee_id="E1001")
    assert r.route == "leave_balance" and "11.5" in r.answer
    assert "retrieve" not in steps(r)


def test_leave_balance_without_id_asks_for_it(assistant):
    r = assistant.ask("What is my leave balance?")
    assert "employee ID" in r.answer


def test_verify_strips_hallucinated_citations(assistant, monkeypatch):
    real = assistant.llm.generate
    monkeypatch.setattr(assistant.llm, "generate", lambda q, ch: real(q, ch) + " Extra claim [9].")
    r = assistant.ask("How long is the probation period?")
    assert "[9]" not in r.answer and not r.refused


def test_answer_without_citation_is_refused(assistant, monkeypatch):
    monkeypatch.setattr(assistant.llm, "generate", lambda q, ch: "Probation is 90 days.")
    r = assistant.ask("How long is the probation period?")
    assert r.refused
