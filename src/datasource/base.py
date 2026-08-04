"""데이터 소스 인터페이스.

리플레이(과거 재생)와 실시간 API가 **같은 인터페이스**를 구현하도록 강제한다.
판단 엔진과 UI는 이 인터페이스만 알고, 뒤에 무엇이 있는지 몰라야 나중에
ZENTRA Cloud API로 갈아끼울 때 판단 로직을 건드리지 않는다.

핵심 불변식 — **미래를 보지 않는다(no lookahead)**:
  history()/latest()/today() 는 어떤 구현이든 now() 이후의 값을 반환해서는 안 된다.
  이게 깨지면 백테스트 결과가 실제보다 좋게 나오고, 그 파라미터로 현장에 나가면
  성능이 재현되지 않는다. tests/test_datasource.py 가 이 불변식을 검사한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class Reading:
    """한 구역의 한 시점 측정값."""

    timestamp: datetime
    zone_id: str
    ppfd: float | None
    source: str                 # measured / estimated_* / missing
    external_ppfd: float | None = None
    air_temperature: float | None = None

    @property
    def is_measured(self) -> bool:
        return self.source == "measured"

    @property
    def is_estimated(self) -> bool:
        return self.source.startswith("estimated")

    @property
    def is_valid(self) -> bool:
        return self.ppfd is not None and not pd.isna(self.ppfd)


class DataSource(ABC):
    """구역별 PPFD 시계열 공급자."""

    interval_minutes: int

    @property
    @abstractmethod
    def zone_ids(self) -> list[str]:
        ...

    @abstractmethod
    def now(self) -> datetime:
        """현재 시각. 리플레이면 가상 시각, 실시간이면 실제 시각."""

    @abstractmethod
    def latest(self, zone_id: str) -> Reading | None:
        """now() 시점(이하)의 가장 최근 측정값."""

    @abstractmethod
    def history(self, zone_id: str, start: datetime | None = None,
                end: datetime | None = None) -> pd.Series:
        """[start, end] 구간의 PPFD 시계열. end 는 now() 를 넘지 못한다."""

    @abstractmethod
    def external_history(self, start: datetime | None = None,
                         end: datetime | None = None) -> pd.Series:
        """외부 PPFD 시계열. 동일하게 now() 를 넘지 못한다."""

    def temperature_history(self, zone_id: str, start: datetime | None = None,
                            end: datetime | None = None) -> pd.Series:
        """기온 시계열. 온도 데이터가 없는 구현은 빈 Series 를 돌려준다."""
        return pd.Series(dtype=float)

    def sensor_age_minutes(self, zone_id: str) -> float | None:
        """마지막 '실측' 이후 경과 분. 센서·통신 이상 감지에 쓴다.

        추정으로 채워진 값은 실측이 아니므로 age 를 리셋하지 않는다 —
        결측을 보정해 놓고 센서가 살아 있다고 착각하면 안 된다.
        """
        return None

    def today(self, zone_id: str) -> pd.Series:
        """당일 00:00 부터 now() 까지."""
        from ..timeutil import day_bounds
        start, _ = day_bounds(self.now())
        return self.history(zone_id, start, self.now())

    def external_today(self) -> pd.Series:
        from ..timeutil import day_bounds
        start, _ = day_bounds(self.now())
        return self.external_history(start, self.now())

    @property
    def interval_seconds(self) -> int:
        return self.interval_minutes * 60
