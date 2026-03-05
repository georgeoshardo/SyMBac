from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class WorkerProgress:
    message: str
    fraction: float | None = None
