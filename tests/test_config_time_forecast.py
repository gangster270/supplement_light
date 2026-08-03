"""설정 검증 / 자정을 넘는 시간창 / 예측기 동작 테스트."""

from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.forecast import (ClearnessDynamics, Forecaster, learn_clearness_dynamics,
                          progress_bin)
from src.metrics import ClearSkyProfile
from src.timeutil import TimeWindow, day_bounds, in_any_window, parse_time, slot_range


class TestTimeWindow:
    def test_보통_구간(self):
        w = TimeWindow.parse("04:00-08:00")
        assert not w.wraps_midnight
        assert w.contains(time(5, 0))
        assert not w.contains(time(9, 0))

    def test_자정을_넘는_구간(self):
        """NI 18:30~02:30 이 제대로 해석되지 않으면 실험 처리 자체가 틀어진다."""
        w = TimeWindow(time(18, 30), time(2, 30), include_start=True, include_end=True)
        assert w.wraps_midnight
        for t in [time(18, 30), time(20, 0), time(23, 59), time(0, 0), time(2, 30)]:
            assert w.contains(t), t
        for t in [time(18, 29), time(2, 31), time(12, 0)]:
            assert not w.contains(t), t

    def test_경계_포함_설정이_반영된다(self):
        closed = TimeWindow(time(4, 0), time(8, 0), include_start=True, include_end=True)
        half = TimeWindow(time(4, 0), time(8, 0), include_start=True, include_end=False)
        assert closed.contains(time(8, 0))
        assert not half.contains(time(8, 0))

    def test_datetime_도_받는다(self):
        w = TimeWindow.parse("04:00-08:00")
        assert w.contains(datetime(2026, 5, 1, 5, 0))

    def test_여러_창_검사(self):
        ws = [TimeWindow.parse("04:00-08:00"), TimeWindow.parse("15:00-22:00")]
        assert in_any_window(datetime(2026, 5, 1, 16, 0), ws)
        assert not in_any_window(datetime(2026, 5, 1, 12, 0), ws)

    def test_시각_파싱(self):
        assert parse_time("18:30") == time(18, 30)
        assert parse_time("18:30:15") == time(18, 30, 15)

    def test_달력일_경계(self):
        s, e = day_bounds(datetime(2026, 5, 15, 20, 0))
        assert s == datetime(2026, 5, 15, 0, 0)
        assert e == datetime(2026, 5, 16, 0, 0)

    def test_슬롯_생성은_반개구간(self):
        slots = slot_range(datetime(2026, 5, 1, 0, 0), datetime(2026, 5, 1, 1, 0), 10)
        assert len(slots) == 6
        assert slots[-1] == datetime(2026, 5, 1, 0, 50)


class TestConfig:
    def test_실제_site_yaml_이_로드된다(self):
        cfg = load_config()
        assert cfg.zone_ids
        assert cfg.interval_seconds == cfg.interval_minutes * 60

    def test_미확정_그룹이_드러난다(self):
        """조용히 기본값으로 굴러가면 왜곡된 줄도 모르고 쓰게 된다."""
        cfg = load_config()
        assert not cfg.is_fully_verified
        assert cfg.unverified_message()

    def test_구역_조회(self):
        cfg = load_config()
        z = cfg.zone(cfg.zone_ids[0])
        assert cfg.greenhouse_of(z.id).id == z.greenhouse_id
        with pytest.raises(KeyError):
            cfg.zone("없는구역")

    def test_기준구역은_온실마다_하나(self):
        cfg = load_config()
        for gh in cfg.greenhouses:
            assert gh.reference_zone in gh.zones
            assert gh.reference_zone not in gh.sibling_zones

    def test_히스테리시스가_뒤집히면_로드_실패(self, tmp_path):
        import yaml
        from src.config import DEFAULT_CONFIG_PATH
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["decision"]["on_threshold"] = 300.0   # off(250) 보다 크게 → 채터링 구조
        p = tmp_path / "config" / "site.yaml"
        p.parent.mkdir(parents=True)
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match="히스테리시스"):
            load_config(p)

    def test_등기구_기여량이_0이면_로드_실패(self, tmp_path):
        """0이면 필요 점등시간이 무한대가 된다. 조용히 넘어가면 안 된다."""
        import yaml
        from src.config import DEFAULT_CONFIG_PATH
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["lamp"]["ppfd_contribution"] = 0.0
        p = tmp_path / "config" / "site.yaml"
        p.parent.mkdir(parents=True)
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        with pytest.raises(ValueError, match="ppfd_contribution"):
            load_config(p)

    def test_NI가_일일상한을_소진하면_경고(self, tmp_path):
        """NI 8시간 + 상한 8시간이면 보광 판단이 항상 상한에 막힌다. 조용히 넘어가면 안 된다."""
        import yaml
        from src.config import DEFAULT_CONFIG_PATH
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw["decision"]["max_daily_lighting_hours"] = 8.0   # NI(약 8.2h) 보다 작다
        p = tmp_path / "config" / "site.yaml"
        p.parent.mkdir(parents=True)
        p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        cfg = load_config(p)
        assert any("항상 상한에 막힙니다" in w for w in cfg.warnings)

    def test_기본_설정은_경고가_없다(self):
        assert load_config().warnings == []

    def test_등기구_시간당_DLI_환산(self):
        from src.config import LampConfig
        lamp = LampConfig(power_w_per_fixture=100.0, fixtures_per_zone=10,
                          ppfd_contribution=100.0)
        assert lamp.dli_per_hour() == pytest.approx(0.36)
        assert lamp.power_kw_per_zone == pytest.approx(1.0)


class TestForecaster:
    def _profile(self, frame):
        return ClearSkyProfile(frame.ppfd["z1"], 10, quantile=0.9, doy_window_days=15)

    def test_일몰_이후는_확실한_0(self, cfg, sample_frame):
        f = Forecaster(self._profile(sample_frame), cfg)
        fc = f.predict(sample_frame.ppfd["z1"]["2026-05-02"], datetime(2026, 5, 2, 23, 0))
        assert fc.expected == 0.0
        assert fc.confidence == "high"       # 불확실한 게 아니라 확실히 없는 것

    def test_CI를_못_구하면_보수적_기본값과_넓은_밴드(self, cfg, sample_frame):
        f = Forecaster(self._profile(sample_frame), cfg, default_clearness=0.7)
        day = sample_frame.ppfd["z1"]["2026-05-02"]
        m = datetime(2026, 5, 2, 3, 0)
        fc = f.predict(day[day.index < pd.Timestamp(m)], m)
        assert fc.clearness is None
        assert fc.confidence == "low"
        assert fc.expected == pytest.approx(fc.clear_sky_remaining * 0.7, rel=1e-6)
        assert fc.band_width > 0

    def test_맑은_날은_청천_잔여에_가깝게_예측(self, cfg, sample_frame):
        f = Forecaster(self._profile(sample_frame), cfg)
        day = sample_frame.ppfd["z1"]["2026-05-02"]
        m = datetime(2026, 5, 2, 12, 0)
        fc = f.predict(day[day.index < pd.Timestamp(m)], m)
        assert fc.expected == pytest.approx(fc.clear_sky_remaining, rel=0.1)

    def test_흐린_날은_비례해서_줄어든다(self, cfg, sample_frame):
        f = Forecaster(self._profile(sample_frame), cfg)
        day = sample_frame.ppfd["z1"]["2026-05-02"]
        m = datetime(2026, 5, 2, 12, 0)
        clear = f.predict(day[day.index < pd.Timestamp(m)], m)
        cloudy = f.predict(day[day.index < pd.Timestamp(m)] * 0.4, m)
        assert cloudy.expected == pytest.approx(clear.expected * 0.4, rel=0.05)

    def test_예측_모드가_반영된다(self, cfg, sample_frame):
        f = Forecaster(self._profile(sample_frame), cfg)
        day = sample_frame.ppfd["z1"]["2026-05-02"]
        m = datetime(2026, 5, 2, 12, 0)
        sofar = day[day.index < pd.Timestamp(m)]
        cons = f.predict(sofar, m, mode="conservative").expected
        std = f.predict(sofar, m, mode="standard").expected
        aggr = f.predict(sofar, m, mode="aggressive").expected
        assert cons < std < aggr    # 보수적일수록 자연광을 적게 잡아 더 자주 점등

    def test_밴드가_예측을_감싼다(self, cfg, sample_frame):
        dyn = ClearnessDynamics(overall=(0.5, 1.5), n_samples=100)
        f = Forecaster(self._profile(sample_frame), cfg, dynamics=dyn)
        day = sample_frame.ppfd["z1"]["2026-05-02"]
        m = datetime(2026, 5, 2, 12, 0)
        fc = f.predict(day[day.index < pd.Timestamp(m)], m)
        assert fc.low < fc.expected < fc.high


class TestClearnessDynamics:
    def test_진행률_구간_분류(self):
        assert progress_bin(0.0) == 0
        assert progress_bin(0.55) == 5
        assert progress_bin(1.0) == 9      # 상한은 마지막 구간으로

    def test_학습_데이터가_없으면_보정_안_함(self):
        assert ClearnessDynamics().calibration(0.5) == 1.0

    def test_실제_학습이_동작한다(self, cfg, sample_frame):
        p = ClearSkyProfile(sample_frame.ppfd["z1"], 10, 0.9, 15)
        dyn = learn_clearness_dynamics(sample_frame, p, "z1", cfg)
        assert dyn.n_samples > 0
        lo, hi = dyn.band(0.5)
        assert lo <= hi
