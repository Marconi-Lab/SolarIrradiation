from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Sequence, Literal
Source = Literal["NASA_POWER", "CAMS"]
@dataclass(frozen=True)
class DateRange:
    start: date
    end: date
@dataclass(frozen=True)
class LocationSpec:
    lat: float
    lon: float
@dataclass(frozen=True)
class VariableSpec:
    name: str
    code: str | None = None
@dataclass(frozen=True)
class FetchPlan:
    source: Source
    date_range: DateRange
    locations: Sequence[LocationSpec]
    variables: Sequence[VariableSpec]
