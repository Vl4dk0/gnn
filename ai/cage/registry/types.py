from typing import TypedDict


class CageStepEvent(TypedDict):
    step: int
    action: str
    u: int | None
    v: int | None
    accepted: bool
    reward: float
    done: bool
    done_reason: str | None
