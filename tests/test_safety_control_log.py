"""안전 조건(S0) / 제어 명령 / 판단 이력 / 시나리오 테스트."""

from __future__ import annotations

import json
from datetime import date, datetime

import pandas as pd
import pytest

from src.control import build_command, to_json
from src.decision import DisplayStatus, Layer, Signal, decide
from src.logbook import LogBook
from src.scenario import SCENARIOS, build_scenario_frame, inject_sensor_fault
from tests.conftest import make_state


# =====================================================================
# S0: 안전 조건
# =====================================================================

class TestHighTemperature:
    def test_고온이면_점등하지_않는다(self, params):
        state = make_state(now=datetime(2026, 5, 15, 16, 0), treatment="무처리",
                           air_temperature=33.0, dli_today=0.0, forecast_remaining=0.0,
                           ppfd_ma=0.0)
        d = decide(state, params)
        assert d.signal is Signal.OFF
        assert d.layer is Layer.HIGH_TEMP
        assert d.status is DisplayStatus.DEFER
        assert d.warning and "고온" in d.warning

    def test_고온_차단은_최소_유지시간을_무시한다(self, params):
        """작물 보호는 채터링 방지보다 우선한다."""
        state = make_state(now=datetime(2026, 5, 15, 16, 0), treatment="무처리",
                           air_temperature=33.0, lamp_on=True, minutes_in_state=5)
        d = decide(state, params)
        assert d.signal is Signal.OFF
        assert d.blocked_by is None

    def test_온도_히스테리시스(self, params):
        """31℃(max 32 / resume 30)에서 켜져 있으면 유지, 꺼져 있으면 재점등 안 함."""
        hot_on = make_state(now=datetime(2026, 5, 15, 16, 0), treatment="무처리",
                            air_temperature=31.0, lamp_on=True, minutes_in_state=60)
        assert decide(hot_on, params).layer is not Layer.HIGH_TEMP   # 32 미만이라 유지

        hot_off = make_state(now=datetime(2026, 5, 15, 16, 0), treatment="무처리",
                             air_temperature=31.0, lamp_on=False, minutes_in_state=60,
                             dli_today=0.0, forecast_remaining=0.0, ppfd_ma=0.0)
        assert decide(hot_off, params).layer is Layer.HIGH_TEMP      # 30 초과라 재점등 금지

    def test_NI는_기본적으로_고온에도_유지된다(self, params):
        """실험 처리 조건을 임의로 깨지 않는다. 대신 경고를 붙인다."""
        state = make_state(now=datetime(2026, 5, 15, 20, 0), treatment="NI",
                           air_temperature=35.0)
        d = decide(state, params)
        assert d.signal is Signal.ON
        assert d.layer is Layer.NI
        assert d.warning and "고온" in d.warning

    def test_설정하면_고온이_NI도_덮는다(self, cfg, params):
        from dataclasses import replace
        p = replace(params, safety=replace(cfg.safety, override_ni_on_high_temp=True))
        state = make_state(now=datetime(2026, 5, 15, 20, 0), treatment="NI",
                           air_temperature=35.0)
        d = decide(state, p)
        assert d.signal is Signal.OFF
        assert d.layer is Layer.HIGH_TEMP
        assert "연구자 확인" in d.reason

    def test_온도_데이터가_없으면_고온판단을_건너뛴다(self, params):
        state = make_state(air_temperature=None)
        assert decide(state, params).layer is not Layer.HIGH_TEMP


class TestSensorFault:
    def test_센서가_오래_갱신되지_않으면_상태_유지(self, params):
        state = make_state(now=datetime(2026, 5, 15, 16, 0), treatment="무처리",
                           sensor_age_minutes=90, lamp_on=False)
        d = decide(state, params)
        assert d.layer is Layer.SENSOR_FAULT
        assert d.status is DisplayStatus.CHECK_SENSOR
        assert d.signal is Signal.OFF          # 현재 상태 유지

    def test_점등_중_센서_이상이면_점등을_유지한다(self, params):
        """임의로 끄면 그 사이 광량이 부족해진다. 사람이 확인할 때까지 유지가 안전하다."""
        state = make_state(now=datetime(2026, 5, 15, 16, 0), treatment="무처리",
                           sensor_age_minutes=90, lamp_on=True)
        assert decide(state, params).signal is Signal.ON

    def test_허용_범위_안이면_정상_판단(self, params):
        state = make_state(treatment="무처리", sensor_age_minutes=20)
        assert decide(state, params).layer is not Layer.SENSOR_FAULT

    def test_NI_구간은_센서와_무관하게_점등(self, params):
        """NI 는 스케줄 기반이라 센서가 죽어도 실행할 수 있다."""
        state = make_state(now=datetime(2026, 5, 15, 20, 0), treatment="NI",
                           sensor_age_minutes=999)
        d = decide(state, params)
        assert d.layer is Layer.NI and d.signal is Signal.ON


class TestDisplayStatus:
    def test_4상태가_모두_나온다(self, params):
        cases = {
            DisplayStatus.RECOMMEND_ON: make_state(now=datetime(2026, 5, 15, 20, 0)),
            DisplayStatus.NOT_NEEDED: make_state(dli_today=30.0, forecast_remaining=0.0),
            DisplayStatus.CHECK_SENSOR: make_state(treatment="무처리",
                                                   sensor_age_minutes=120),
            DisplayStatus.DEFER: make_state(treatment="무처리", air_temperature=40.0),
        }
        for expected, state in cases.items():
            assert decide(state, params).status is expected, expected


# =====================================================================
# 제어 명령
# =====================================================================

class TestControlCommand:
    def test_JSON_직렬화가_된다(self, params):
        state = make_state()
        d = decide(state, params)
        cmd = build_command(state, d, "1구역", "1온실")
        parsed = json.loads(to_json(cmd))
        assert parsed["command"]["action"] in ("ON", "OFF")
        assert parsed["target"]["zone_id"] == state.zone_id

    def test_항상_dry_run(self, params):
        """시제품이 실제 하드웨어를 건드리면 안 된다."""
        state = make_state()
        cmd = build_command(state, decide(state, params))
        assert cmd["dry_run"] is True

    def test_근거가_명령에_포함된다(self, params):
        state = make_state()
        cmd = build_command(state, decide(state, params))
        assert cmd["rationale"]["reason"]
        assert cmd["rationale"]["layer"]
        assert cmd["rationale"]["status"]


# =====================================================================
# 판단 이력
# =====================================================================

class TestLogBook:
    def test_기록과_CSV_변환(self, params):
        lb = LogBook()
        for hour in (16, 17, 20):
            s = make_state(now=datetime(2026, 5, 15, hour, 0))
            lb.append(s, decide(s, params), "1구역")
        assert len(lb) == 3
        df = lb.to_frame()
        assert len(df) == 3 and "근거" in df.columns
        raw = lb.to_csv_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")     # 엑셀 한글 깨짐 방지 BOM

    def test_빈_이력도_컬럼을_갖는다(self):
        assert list(LogBook().to_frame().columns)

    def test_요약_계산(self, params):
        lb = LogBook()
        for i in range(6):                          # 6스텝 = 60분
            s = make_state(now=datetime(2026, 5, 15, 20, i * 10))
            lb.append(s, decide(s, params), "1구역")
        summary = lb.summary(interval_minutes=10, power_kw=1.0)
        assert summary["steps"] == 6
        assert summary["점등시간"] == pytest.approx(1.0)   # NI 구간이라 전부 점등
        assert summary["kWh"] == pytest.approx(1.0)
        assert summary["요금"] > 0


# =====================================================================
# 시나리오
# =====================================================================

class TestScenario:
    def test_모든_시나리오가_생성된다(self, cfg):
        for key in SCENARIOS:
            frame = build_scenario_frame(cfg, key, date(2026, 5, 20))
            assert len(frame.index) == 24 * 6
            assert frame.has_temperature
            assert set(frame.zone_ids) == set(cfg.zone_ids)

    def test_흐림이_맑음보다_어둡다(self, cfg):
        from src.metrics import dli_from_ppfd
        clear = build_scenario_frame(cfg, "clear", date(2026, 5, 20))
        cloudy = build_scenario_frame(cfg, "overcast", date(2026, 5, 20))
        z = cfg.zone_ids[0]
        assert (dli_from_ppfd(cloudy.ppfd[z], 600)
                < dli_from_ppfd(clear.ppfd[z], 600) * 0.6)

    def test_고온_시나리오가_한계를_넘는다(self, cfg):
        frame = build_scenario_frame(cfg, "heatwave", date(2026, 5, 20))
        assert frame.temperature[cfg.zone_ids[0]].max() > cfg.safety.max_air_temperature

    def test_센서_두절_주입(self, cfg):
        frame = build_scenario_frame(cfg, "clear", date(2026, 5, 20))
        z = cfg.zone_ids[0]
        broken = inject_sensor_fault(frame, z, datetime(2026, 5, 20, 11, 0), 90)
        window = broken.ppfd[z]["2026-05-20 11:00":"2026-05-20 12:20"]
        assert window.isna().all()
        assert (broken.source[z]["2026-05-20 11:00":"2026-05-20 12:20"] == "missing").all()


class TestSafetyEndToEnd:
    def test_센서_두절이_실제로_센서확인_상태를_만든다(self, cfg):
        """주입한 결측이 datasource → engine → decide 를 거쳐 화면 상태까지 이어지는지."""
        from src.datasource.replay import ReplayDataSource
        from src.engine import DecisionEngine, prepare
        frame = build_scenario_frame(cfg, "overcast", date(2026, 5, 20))
        z = cfg.zone_ids[1]                     # NI 비대상이 아니어도 낮 시간이면 발동
        broken = inject_sensor_fault(frame, z, datetime(2026, 5, 20, 10, 0), 120)
        profiles, forecasters = prepare(cfg, frame)
        engine = DecisionEngine(cfg, profiles, forecasters)
        source = ReplayDataSource(broken, start=datetime(2026, 5, 20, 11, 50))
        _, decision = engine.evaluate(source, z)
        assert decision.layer is Layer.SENSOR_FAULT
        assert decision.status is DisplayStatus.CHECK_SENSOR
