"""P2 지표 계산 테스트. 손으로 검산 가능한 값으로 검증한다."""

from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd
import pytest

from src.metrics import (ClearSkyProfile, clearness_index, cumulative_dli, daily_dli,
                         dli_from_ppfd, hourly_transmittance, latest_moving_average,
                         moving_average, night_baseline, remove_lamp_contribution,
                         sibling_ratio)
from src.timeutil import TimeWindow


class TestDLI:
    def test_손계산과_일치한다(self):
        """PPFD 100 µmol 을 10분(600초) → 100×600/1e6 = 0.06 mol."""
        s = pd.Series([100.0], index=pd.DatetimeIndex(["2026-05-01 12:00"]))
        assert dli_from_ppfd(s, 600) == pytest.approx(0.06)

    def test_1시간_100umol_은_0_36mol(self):
        idx = pd.date_range("2026-05-01 12:00", periods=6, freq="10min")
        assert dli_from_ppfd(pd.Series([100.0] * 6, index=idx), 600) == pytest.approx(0.36)

    def test_결측은_0이_아니라_건너뛴다(self):
        idx = pd.date_range("2026-05-01 12:00", periods=3, freq="10min")
        s = pd.Series([100.0, np.nan, 100.0], index=idx)
        assert dli_from_ppfd(s, 600) == pytest.approx(0.12)

    def test_빈_시계열은_0(self):
        assert dli_from_ppfd(pd.Series(dtype=float), 600) == 0.0

    def test_누적_DLI는_단조증가(self, sample_frame):
        c = cumulative_dli(sample_frame.ppfd["z1"], 600)
        assert (c.diff().dropna() >= -1e-12).all()
        assert c.iloc[-1] == pytest.approx(dli_from_ppfd(sample_frame.ppfd["z1"], 600))

    def test_일별_DLI는_달력일_기준(self, sample_frame):
        d = daily_dli(sample_frame.ppfd["z1"], 600)
        assert len(d) == 3


class TestMovingAverage:
    def test_창_길이가_간격으로_환산된다(self):
        idx = pd.date_range("2026-05-01 00:00", periods=6, freq="10min")
        s = pd.Series([0, 0, 0, 300, 300, 300], index=idx, dtype=float)
        ma = moving_average(s, window_minutes=30, interval_minutes=10)
        assert ma.iloc[-1] == pytest.approx(300.0)
        assert ma.iloc[3] == pytest.approx(100.0)   # (0+0+300)/3

    def test_최근_이동평균만_뽑기(self):
        idx = pd.date_range("2026-05-01 00:00", periods=6, freq="10min")
        s = pd.Series([0, 0, 0, 0, 0, 300], index=idx, dtype=float)
        assert latest_moving_average(s, 30, 10) == pytest.approx(100.0)

    def test_전부_결측이면_nan(self):
        idx = pd.date_range("2026-05-01 00:00", periods=3, freq="10min")
        assert np.isnan(latest_moving_average(pd.Series([np.nan] * 3, index=idx), 30, 10))


class TestTransmittance:
    def test_시간대별_투과율을_복원한다(self):
        """외부 1000, 내부 500 이면 투과율 0.5 가 나와야 한다."""
        idx = pd.date_range("2026-05-01 00:00", periods=144, freq="10min")
        ext = pd.Series(np.where((idx.hour >= 8) & (idx.hour < 16), 1000.0, 0.0), index=idx)
        internal = ext * 0.5
        ratios = hourly_transmittance(internal, ext, daylight_threshold=50.0)
        for h in range(8, 16):
            assert ratios[h] == pytest.approx(0.5)

    def test_어두운_시점은_노이즈로_배제된다(self):
        idx = pd.date_range("2026-05-01 00:00", periods=6, freq="10min")
        ext = pd.Series([10.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0], index=idx)
        internal = pd.Series([9.0, 9.0, 9.0, 400.0, 400.0, 400.0], index=idx)
        # 임계값 50 미만인 앞 3개(비율 0.9)가 섞이면 중앙값이 오염된다
        ratios = hourly_transmittance(internal, ext, daylight_threshold=50.0)
        assert ratios[0] == pytest.approx(0.4)

    def test_형제구역_비율은_합계비(self):
        idx = pd.date_range("2026-05-01 12:00", periods=3, freq="10min")
        ref = pd.Series([100.0, 200.0, 300.0], index=idx)
        sib = pd.Series([90.0, 180.0, 270.0], index=idx)
        assert sibling_ratio(ref, sib, reference_threshold=50.0) == pytest.approx(0.9)


class TestNightBaseline:
    def test_순수_야간만_사용한다(self):
        """시각상 야간이어도 외부광이 있으면 제외해야 한다 (하지 계절 오염 방지)."""
        idx = pd.DatetimeIndex(["2026-06-21 19:00", "2026-06-21 19:10", "2026-06-21 23:00"])
        zone = pd.Series([200.0, 200.0, 60.0], index=idx)      # 앞 2개는 자연광 섞임
        ext = pd.Series([300.0, 300.0, 0.0], index=idx)
        window = TimeWindow(time(18, 30), time(2, 30), True, True)
        assert night_baseline(zone, ext, window, 2.0) == pytest.approx(60.0)

    def test_해당_시점이_없으면_nan(self):
        idx = pd.DatetimeIndex(["2026-05-01 12:00"])
        window = TimeWindow(time(18, 30), time(2, 30), True, True)
        assert np.isnan(night_baseline(pd.Series([100.0], index=idx),
                                       pd.Series([1000.0], index=idx), window, 2.0))


class TestLampRemoval:
    def test_점등_구간에서만_빼고_0에서_자른다(self):
        idx = pd.DatetimeIndex(["2026-05-01 12:00", "2026-05-01 20:00", "2026-05-01 21:00"])
        s = pd.Series([500.0, 60.0, 30.0], index=idx)
        window = TimeWindow(time(18, 30), time(2, 30), True, True)
        out = remove_lamp_contribution(s, [window], 60.0)
        assert out.iloc[0] == pytest.approx(500.0)   # 주간은 그대로
        assert out.iloc[1] == pytest.approx(0.0)     # 60 - 60
        assert out.iloc[2] == pytest.approx(0.0)     # 음수는 0으로

    def test_점등창이_없으면_원본_그대로(self):
        idx = pd.DatetimeIndex(["2026-05-01 20:00"])
        s = pd.Series([60.0], index=idx)
        assert remove_lamp_contribution(s, [], 60.0).iloc[0] == 60.0


class TestClearSkyProfile:
    def test_상위분위수가_기준선이_된다(self, sample_frame):
        p = ClearSkyProfile(sample_frame.ppfd["z1"], 10, quantile=0.9, doy_window_days=15)
        prof = p.for_date(datetime(2026, 5, 2))
        assert prof.max() > 0
        assert prof.loc[0] == pytest.approx(0.0)        # 자정은 0
        assert prof.idxmax() == 750                     # 12:30 피크

    def test_구간_적산이_전체_적산과_맞는다(self, sample_frame):
        p = ClearSkyProfile(sample_frame.ppfd["z1"], 10, 0.9, 15)
        d = datetime(2026, 5, 2)
        whole = p.dli_between(d, 0, 1440)
        halves = p.dli_between(d, 0, 720) + p.dli_between(d, 720, 1440)
        assert whole == pytest.approx(halves)

    def test_경과와_잔여의_합이_하루_전체(self, sample_frame):
        p = ClearSkyProfile(sample_frame.ppfd["z1"], 10, 0.9, 15)
        m = datetime(2026, 5, 2, 14, 0)
        assert p.elapsed_dli(m) + p.remaining_dli(m) == pytest.approx(p.dli_between(m, 0, 1440))

    def test_역구간은_0(self, sample_frame):
        p = ClearSkyProfile(sample_frame.ppfd["z1"], 10, 0.9, 15)
        assert p.dli_between(datetime(2026, 5, 2), 800, 700) == 0.0


class TestClearnessIndex:
    def test_청천과_같으면_1에_가깝다(self, sample_frame):
        p = ClearSkyProfile(sample_frame.ppfd["z1"], 10, quantile=0.9, doy_window_days=15)
        day = sample_frame.ppfd["z1"]["2026-05-02"]
        m = datetime(2026, 5, 2, 15, 0)
        ci = clearness_index(day[day.index < pd.Timestamp(m)], p, m, 600)
        assert ci == pytest.approx(1.0, abs=0.05)

    def test_광량이_절반이면_0_5(self, sample_frame):
        p = ClearSkyProfile(sample_frame.ppfd["z1"], 10, quantile=0.9, doy_window_days=15)
        day = sample_frame.ppfd["z1"]["2026-05-02"] * 0.5
        m = datetime(2026, 5, 2, 15, 0)
        ci = clearness_index(day[day.index < pd.Timestamp(m)], p, m, 600)
        assert ci == pytest.approx(0.5, abs=0.05)

    def test_기준이_너무_작으면_None(self, sample_frame):
        """이른 새벽에는 분모가 작아 비율이 폭발한다. 계산하지 않는 게 맞다."""
        p = ClearSkyProfile(sample_frame.ppfd["z1"], 10, 0.9, 15)
        m = datetime(2026, 5, 2, 3, 0)
        day = sample_frame.ppfd["z1"]["2026-05-02"]
        assert clearness_index(day[day.index < pd.Timestamp(m)], p, m, 600) is None
