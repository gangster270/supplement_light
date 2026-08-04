"""P4 런타임(engine) 테스트 — 상태 추적과 반사실 처리."""

from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd
import pytest

from src.datasource.replay import ReplayDataSource
from src.decision import Signal
from src.engine import DecisionEngine, ZoneRuntime, prepare
from src.metrics import ClearSkyProfile
from src.timeutil import TimeWindow


@pytest.fixture
def engine(cfg, sample_frame):
    profiles, forecasters = prepare(cfg, sample_frame)
    return DecisionEngine(cfg, profiles, forecasters)


class TestZoneRuntime:
    def test_상태_전환시_유지시간이_초기화된다(self):
        rt = ZoneRuntime(lamp_on=False, minutes_in_state=120)
        rt.apply(datetime(2026, 5, 1, 12, 0), Signal.ON, 10, 0.0, 10.0)
        assert rt.lamp_on and rt.minutes_in_state == 0
        rt.apply(datetime(2026, 5, 1, 12, 10), Signal.ON, 10, 0.0, 10.0)
        assert rt.minutes_in_state == 10

    def test_점등시간이_누적된다(self):
        rt = ZoneRuntime()
        for i in range(6):
            rt.apply(datetime(2026, 5, 1, 12, i * 10), Signal.ON, 10, 0.0, 10.0)
        assert rt.lighting_minutes_today == 60

    def test_날짜가_바뀌면_일일_누적이_초기화된다(self):
        rt = ZoneRuntime(lighting_minutes_today=300, current_day=datetime(2026, 5, 1).date())
        rt.roll_day(datetime(2026, 5, 2).date())
        assert rt.lighting_minutes_today == 0

    def test_광주기는_자연광이_있어도_누적된다(self):
        rt = ZoneRuntime()
        rt.apply(datetime(2026, 5, 1, 12, 0), Signal.OFF, 10, 500.0, 10.0)
        assert rt.photoperiod_minutes_today == 10   # 소등이지만 자연광이 밝다


class TestCounterfactual:
    """리플레이 데이터의 과거 점등분을 빼지 않으면 DLI가 이중 계산된다."""

    def test_과거_점등분이_제거된다(self, cfg, sample_frame):
        profiles, forecasters = prepare(cfg, sample_frame)
        eng = DecisionEngine(cfg, profiles, forecasters)
        idx = pd.DatetimeIndex(["2026-05-01 12:00", "2026-05-01 20:00"])
        measured = pd.Series([500.0, 100.0], index=idx)
        natural = eng.natural_series("z1", measured)
        assert natural.iloc[0] == pytest.approx(500.0)    # 주간은 그대로
        assert natural.iloc[1] == pytest.approx(0.0)      # NI 구간: 100 - 100

    def test_NI_비대상_구역은_그대로(self, cfg, sample_frame):
        profiles, forecasters = prepare(cfg, sample_frame)
        eng = DecisionEngine(cfg, profiles, forecasters,
                             historical_lighting_windows={"z1": [], "z2": []})
        idx = pd.DatetimeIndex(["2026-05-01 20:00"])
        assert eng.natural_series("z1", pd.Series([100.0], index=idx)).iloc[0] == 100.0

    def test_우리_점등분이_DLI에_반영된다(self, cfg, sample_frame, engine):
        src = ReplayDataSource(sample_frame, start=datetime(2026, 5, 2, 16, 0))
        before = engine.build_state(src, "z1").dli_today
        engine.runtime["z1"].roll_day(datetime(2026, 5, 2).date())
        engine.runtime["z1"].lighting_minutes_today = 60      # 1시간 켰다고 가정
        after = engine.build_state(src, "z1").dli_today
        assert after - before == pytest.approx(cfg.lamp.dli_per_hour(), abs=1e-9)


class TestEngineLoop:
    def test_전_구역이_판단된다(self, engine, sample_frame):
        src = ReplayDataSource(sample_frame, start=datetime(2026, 5, 2, 12, 0))
        out = engine.step(src)
        assert set(out) == {"z1", "z2"}
        for state, decision in out.values():
            assert decision.reason

    def test_하루를_돌려도_상한을_넘지_않는다(self, engine, sample_frame, cfg):
        """엔진이 스스로 만든 상태로 상한을 지키는지 — 순수 함수 테스트로는 못 잡는 부분."""
        src = ReplayDataSource(sample_frame)
        for s in src.iter_steps(datetime(2026, 5, 2), datetime(2026, 5, 3)):
            engine.step(s)
        for zone_id, rt in engine.runtime.items():
            assert rt.lighting_minutes_today <= cfg.decision.max_daily_lighting_hours * 60 + 1e-9

    def test_NI_구간에는_반드시_점등된다(self, engine, sample_frame):
        src = ReplayDataSource(sample_frame)
        ni_signals = []
        for s in src.iter_steps(datetime(2026, 5, 2), datetime(2026, 5, 3)):
            out = engine.step(s)
            if engine.cfg.ni.window.contains(s.now()):
                ni_signals.append(out["z1"][1].signal)
        assert ni_signals and all(sig is Signal.ON for sig in ni_signals)

    def test_evaluate_는_상태를_바꾸지_않는다(self, engine, sample_frame):
        """UI가 매초 다시 그려도 운전 상태가 흘러가면 안 된다."""
        src = ReplayDataSource(sample_frame, start=datetime(2026, 5, 2, 20, 0))
        before = dict(vars(engine.runtime["z1"]))
        for _ in range(5):
            engine.evaluate(src, "z1")
        assert dict(vars(engine.runtime["z1"])) == before

    def test_판단_중_미래를_보지_않는다(self, engine, sample_frame):
        src = ReplayDataSource(sample_frame)
        for s in src.iter_steps(datetime(2026, 5, 2), datetime(2026, 5, 2, 6, 0)):
            state = engine.build_state(s, "z1")
            assert state.now == s.now()
            assert s.today("z1").index.max() <= pd.Timestamp(s.now())


class TestPrepare:
    def test_학습구간을_자른다(self, cfg, sample_frame):
        """평가 구간 데이터로 프로파일을 학습하면 미래를 본 셈이 된다."""
        profiles, _ = prepare(cfg, sample_frame, training_end=datetime(2026, 5, 2))
        assert profiles["z1"].for_date(datetime(2026, 5, 2)).max() > 0

    def test_빈_학습구간은_에러(self, cfg, sample_frame):
        with pytest.raises(ValueError, match="학습 구간"):
            prepare(cfg, sample_frame, training_end=datetime(2020, 1, 1))
