"""Base class and result container for ingest jobs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..io.bq import BigQueryClient
from .types import FetchPlan


@dataclass(frozen=True)
class JobResult:
    """Outcome of an ingest job run.

    Carries observability counters per CLAUDE.md "ingest constraints" memo:
    rows added vs already cached vs skipped, API calls made, when started
    and finished. Logged at the end of every ``run()``.
    """

    job_name: str
    plan_summary: str
    started_at: datetime
    finished_at: datetime
    rows_added: int = 0
    rows_already_cached: int = 0
    rows_skipped: int = 0
    api_calls_made: int = 0
    extra: dict[str, int] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    def summary(self) -> str:
        parts = [
            f"job={self.job_name}",
            f"plan={self.plan_summary}",
            f"rows_added={self.rows_added}",
            f"already_cached={self.rows_already_cached}",
            f"skipped={self.rows_skipped}",
            f"api_calls={self.api_calls_made}",
            f"duration={self.duration_seconds:.1f}s",
        ]
        for k, v in self.extra.items():
            parts.append(f"{k}={v}")
        return " | ".join(parts)


class BaseJob(ABC):
    """Base class for warehouse-population jobs.

    Each subclass owns one ingest pattern (named-location satellite, grid
    satellite, ground-file). The ``run`` method is the only public entry
    point and must return a :class:`JobResult`.
    """

    def __init__(self, bq: BigQueryClient) -> None:
        self._bq = bq

    @property
    def bq(self) -> BigQueryClient:
        return self._bq

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this job (used in logs / job results)."""

    @abstractmethod
    def run(self, plan: FetchPlan) -> JobResult:
        """Execute the ingest. Subclasses dispatch on the concrete plan type."""

    def _start_time(self) -> datetime:
        return datetime.now(timezone.utc)
