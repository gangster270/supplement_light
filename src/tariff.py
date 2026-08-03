"""계시별 전기요금 조회.

P4에서는 판단 엔진의 Layer 4(저가 시간대 배치)에만 쓴다.
청구서 수준의 정산(기본요금·역률·계약전력 등)은 P7 economics.py 에서 다룬다.
"""

from __future__ import annotations

from datetime import datetime

from .config import TariffConfig, TariffSeason, TariffSlot


class TariffLookup:
    def __init__(self, cfg: TariffConfig):
        self.cfg = cfg
        self._by_month: dict[int, TariffSeason] = {}
        for season in cfg.seasons:
            for m in season.months:
                self._by_month[m] = season

    def season_of(self, moment: datetime) -> TariffSeason | None:
        return self._by_month.get(moment.month)

    def slot_of(self, moment: datetime) -> TariffSlot | None:
        season = self.season_of(moment)
        if season is None:
            return None
        for slot in season.slots:
            if slot.window.contains(moment):
                return slot
        return None

    def price(self, moment: datetime) -> float:
        """해당 시각의 단가(원/kWh). 정의되지 않은 시간대는 0 이 아니라 평균 단가로 메운다.

        0을 돌려주면 판단 엔진이 그 시간대를 '공짜'로 보고 몰아서 점등한다.
        """
        slot = self.slot_of(moment)
        if slot is not None:
            return slot.won_per_kwh
        return self.average_price()

    def average_price(self) -> float:
        prices = [s.won_per_kwh for season in self.cfg.seasons for s in season.slots]
        return float(sum(prices) / len(prices)) if prices else 0.0

    def slot_name(self, moment: datetime) -> str:
        slot = self.slot_of(moment)
        return slot.name if slot else "미정의"

    def cost(self, moment: datetime, kw: float, hours: float) -> float:
        """해당 시각 단가로 kw × hours 의 요금(원)."""
        return self.price(moment) * kw * hours
