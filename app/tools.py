"""Tools the agent can call. The HRIS here is a mock with fixture data;
in a real deployment this would call the HR system's API with the caller's
own auth token, never a shared service account."""

from dataclasses import dataclass

_MOCK_HRIS = {
    "E1001": {"name": "Asha", "annual_leave_left": 11.5, "sick_leave_left": 9, "floating_holidays_left": 1},
    "E1002": {"name": "Ravi", "annual_leave_left": 3.0, "sick_leave_left": 12, "floating_holidays_left": 2},
}


@dataclass
class ToolResult:
    ok: bool
    data: dict | None = None
    error: str | None = None


def get_leave_balance(employee_id: str | None) -> ToolResult:
    if not employee_id:
        return ToolResult(ok=False, error="missing_employee_id")
    record = _MOCK_HRIS.get(employee_id)
    if record is None:
        return ToolResult(ok=False, error="unknown_employee")
    return ToolResult(ok=True, data=record)
