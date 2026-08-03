"""P4 런타임 글루: DataSource → DecisionState → decide().

여기서 다루는 미묘하지만 중요한 문제 두 가지.

1) **반사실(counterfactual) 처리**
   리플레이 데이터에 기록된 PPFD 는 '과거에 실제로 그렇게 운영했을 때'의 값이다.
   우리 판단 엔진이 과거와 다른 시각에 점등하기로 해도 기록된 값은 바뀌지 않는다.
   그대로 쓰면 "과거의 보광 + 우리 보광"이 이중 계산되어 DLI가 부풀려지고,
   백테스트가 실제보다 좋게 나온다.
   → 측정값에서 **과거 점등 기여분을 빼서 '자연광만' 시계열을 복원**하고,
     거기에 **우리 판단으로 켠 시간의 기여분만** 더한다.

   과거 점등 시각은 NI 스케줄로 추정한다 (NI 처리구는 NI 구간에 켜져 있었다).
   NI 구간 밖에 별도 보광이 있었다면 이 추정은 불완전하다 — 그 경우
   `historical_lighting_windows` 로 실제 운영 시간대를 넘겨야 한다.

2) **상태 추적**
   최소 유지시간·일일 점등상한은 '지금까지 우리가 어떻게 켰는지'에 의존한다.
   판단 엔진은 순수 함수이므로, 그 상태는 여기서 들고 있는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from .config import SiteConfig
from .datasource.base import DataSource
from .decision import Decision, DecisionParams, DecisionState, Signal, build_params, decide
from .forecast import Forecaster
from .metrics import ClearSkyProfile, dli_from_ppfd, latest_moving_average
from .metrics import historical_lighting_windows as historical_lighting_windows_default
from .metrics import natural_light_frame, remove_lamp_contribution
from .timeutil import TimeWindow, day_bounds, minute_of_day


@dataclass
class ZoneRuntime:
    """한 구역의 등기구 운전 상태. 판단이 아니라 '상태'만 들고 있다."""

    lamp_on: bool = False
    minutes_in_state: int = 0
    lighting_minutes_today: float = 0.0
    photoperiod_minutes_today: float = 0.0
    current_day: date | None = None
    history: list[tuple[datetime, bool]] = field(default_factory=list)

    def roll_day(self, day: date) -> None:
        if self.current_day != day:
            self.current_day = day
            self.lighting_minutes_today = 0.0
            self.photoperiod_minutes_today = 0.0

    def daily_counters(self, day: date) -> tuple[float, float]:
        """(점등시간, 광주기) 를 **상태 변경 없이** 읽는다.

        날짜가 바뀐 뒤 아직 스텝이 진행되지 않았다면 0으로 본다. 조회(UI 새로고침)가
        운전 상태를 건드리면, 화면을 다시 그렸다는 이유만으로 이력이 달라진다.
        """
        if self.current_day != day:
            return 0.0, 0.0
        return self.lighting_minutes_today, self.photoperiod_minutes_today

    def apply(self, moment: datetime, signal: Signal, interval_minutes: int,
              natural_ppfd: float | None, photoperiod_threshold: float) -> None:
        """한 스텝 진행. 이 스텝 동안 등기구가 켜져 있었다고 보고 누적한다."""
        self.roll_day(moment.date())
        on = signal is Signal.ON
        if on != self.lamp_on:
            self.lamp_on = on
            self.minutes_in_state = 0
        else:
            self.minutes_in_state += interval_minutes
        if on:
            self.lighting_minutes_today += interval_minutes
        lit = on or (natural_ppfd is not None and natural_ppfd > photoperiod_threshold)
        if lit:
            self.photoperiod_minutes_today += interval_minutes
        self.history.append((moment, on))


class DecisionEngine:
    """구역별로 상태를 들고 있으면서 매 스텝 판단을 내는 러너."""

    def __init__(self, cfg: SiteConfig, profiles: dict[str, ClearSkyProfile],
                 forecasters: dict[str, Forecaster],
                 historical_lighting_windows: dict[str, list[TimeWindow]] | None = None,
                 photoperiod_threshold: float = 10.0):
        self.cfg = cfg
        self.params: DecisionParams = build_params(cfg)
        self.profiles = profiles
        self.forecasters = forecasters
        self.runtime: dict[str, ZoneRuntime] = {z.id: ZoneRuntime() for z in cfg.zones}
        self.photoperiod_threshold = photoperiod_threshold
        # 과거 실제 점등 시간대. 기본은 NI 스케줄(해당 처리구에 한해).
        self.historical_lighting_windows = (historical_lighting_windows
                                            or historical_lighting_windows_default(cfg))

    # ----------------------------------------------------------------
    # 자연광 복원
    # ----------------------------------------------------------------
    def natural_series(self, zone_id: str, measured: pd.Series) -> pd.Series:
        """측정값에서 과거 점등 기여분을 빼 자연광만 남긴다 (0 미만은 0으로)."""
        return remove_lamp_contribution(
            measured, self.historical_lighting_windows.get(zone_id, []),
            self.cfg.lamp.ppfd_contribution)

    def simulated_lamp_dli_today(self, zone_id: str, now: datetime) -> float:
        """우리 판단으로 오늘 켠 시간의 DLI 기여분."""
        lighting, _ = self.runtime[zone_id].daily_counters(now.date())
        return (lighting / 60.0) * self.cfg.lamp.dli_per_hour()

    def clear_sky_now(self, zone_id: str, now: datetime) -> float:
        profile = self.profiles.get(zone_id)
        if profile is None:
            return 0.0
        series = profile.for_date(now)
        mod = minute_of_day(now)
        grid = int(mod // self.cfg.interval_minutes) * self.cfg.interval_minutes
        if grid in series.index:
            return float(series.loc[grid])
        nearest = min(series.index, key=lambda m: abs(m - mod))
        return float(series.loc[nearest])

    # ----------------------------------------------------------------
    # 상태 구성 + 판단
    # ----------------------------------------------------------------
    def build_state(self, source: DataSource, zone_id: str) -> DecisionState:
        now = source.now()
        zone = self.cfg.zone(zone_id)
        rt = self.runtime[zone_id]
        lighting_today, photoperiod_today = rt.daily_counters(now.date())

        measured_today = source.today(zone_id)
        natural_today = self.natural_series(zone_id, measured_today)

        # 오늘 실제로 받은 광량 = 자연광 + 우리가 켠 보광
        natural_dli = dli_from_ppfd(natural_today, source.interval_seconds)
        dli_today = natural_dli + self.simulated_lamp_dli_today(zone_id, now)

        natural_ma = latest_moving_average(
            natural_today, self.cfg.decision.moving_average_minutes, self.cfg.interval_minutes)
        # 판단 엔진은 '등기구 기여 포함 실측'을 기대하므로, 우리 등기구 상태를 반영해 되돌린다.
        ppfd_ma = (natural_ma + self.cfg.lamp.ppfd_contribution
                   if rt.lamp_on and not np.isnan(natural_ma) else natural_ma)

        forecaster = self.forecasters.get(zone_id)
        if forecaster is not None:
            fc = forecaster.predict(natural_today, now)
            remaining, low, high = fc.expected, fc.low, fc.high
            note, confidence = fc.note, fc.confidence
        else:
            remaining = low = high = 0.0
            note, confidence = "예측기 없음", "low"

        latest = source.latest(zone_id)
        return DecisionState(
            now=now,
            zone_id=zone_id,
            treatment=zone.treatment,
            ppfd_ma=None if np.isnan(ppfd_ma) else float(ppfd_ma),
            dli_today=dli_today,
            forecast_remaining=remaining,
            forecast_low=low,
            forecast_high=high,
            forecast_note=note,
            forecast_confidence=confidence,
            lamp_on=rt.lamp_on,
            minutes_in_state=rt.minutes_in_state,
            lighting_minutes_today=lighting_today,
            photoperiod_minutes_today=photoperiod_today,
            clear_sky_ppfd=self.clear_sky_now(zone_id, now),
            ppfd_now=latest.ppfd if latest else None,
            ppfd_source=latest.source if latest else "missing",
        )

    def evaluate(self, source: DataSource, zone_id: str) -> tuple[DecisionState, Decision]:
        """판단만 하고 상태는 바꾸지 않는다 (UI 표시용)."""
        state = self.build_state(source, zone_id)
        return state, decide(state, self.params)

    def step(self, source: DataSource) -> dict[str, tuple[DecisionState, Decision]]:
        """전 구역을 판단하고 그 결과로 상태를 한 스텝 진행시킨다 (백테스트/실운전용)."""
        out = {}
        for zone in self.cfg.zones:
            state, decision = self.evaluate(source, zone.id)
            natural = decision.natural_ppfd_estimate
            self.runtime[zone.id].apply(state.now, decision.signal,
                                        self.cfg.interval_minutes, natural,
                                        self.photoperiod_threshold)
            out[zone.id] = (state, decision)
        return out


def prepare(cfg: SiteConfig, frame, training_end: datetime | None = None,
            lighting_windows: dict[str, list[TimeWindow]] | None = None
            ) -> tuple[dict[str, ClearSkyProfile], dict[str, Forecaster]]:
    """청천 프로파일과 예측기를 준비한다 (앱/백테스트 시작 시 1회).

    ★ 학습 전에 **점등 기여분을 제거**한다. 야간 NI 조명이 섞인 채로 학습하면
      청천 기준선과 청천지수가 모두 오염되어 예측이 크게 낙관적으로 치우친다.

    training_end 를 주면 그 이전 데이터로만 학습한다. 백테스트에서 평가 구간의
    데이터로 프로파일을 학습하면 미래를 본 셈이 되므로, 평가 구간 시작 시각을 넘겨야 한다.
    """
    from .forecast import build_forecasters
    from .metrics import build_zone_profiles

    training = frame.slice(end=training_end) if training_end is not None else frame
    if len(training.index) == 0:
        raise ValueError("학습 구간이 비어 있습니다. training_end 를 더 뒤로 잡으세요.")
    natural = natural_light_frame(training, cfg, lighting_windows)
    profiles = build_zone_profiles(natural, cfg)
    forecasters = build_forecasters(natural, profiles, cfg)
    return profiles, forecasters
