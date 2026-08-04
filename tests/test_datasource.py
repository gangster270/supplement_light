"""P1 데이터 계층 테스트. 최우선 검증 대상은 **미래를 보지 않는가**이다."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.datasource.replay import ReplayDataSource
from src.ingest import SiteFrame, build_grid, find_column, logger_id_of


class TestNoLookahead:
    """이 불변식이 깨지면 백테스트가 실제보다 좋게 나오고 현장에서 재현되지 않는다."""

    def test_history_는_현재를_넘지_않는다(self, sample_frame):
        src = ReplayDataSource(sample_frame, start=datetime(2026, 5, 2, 12, 0))
        h = src.history("z1")
        assert h.index.max() <= pd.Timestamp(src.now())

    def test_end_를_미래로_줘도_잘린다(self, sample_frame):
        src = ReplayDataSource(sample_frame, start=datetime(2026, 5, 2, 12, 0))
        h = src.history("z1", end=datetime(2026, 5, 3, 23, 0))
        assert h.index.max() <= pd.Timestamp(src.now())

    def test_외부데이터도_잘린다(self, sample_frame):
        src = ReplayDataSource(sample_frame, start=datetime(2026, 5, 2, 12, 0))
        assert src.external_history().index.max() <= pd.Timestamp(src.now())

    def test_today_는_당일_00시부터_현재까지(self, sample_frame):
        src = ReplayDataSource(sample_frame, start=datetime(2026, 5, 2, 12, 0))
        t = src.today("z1")
        assert t.index.min() == pd.Timestamp("2026-05-02 00:00")
        assert t.index.max() == pd.Timestamp("2026-05-02 12:00")

    def test_iter_steps_전_구간에서_불변식이_유지된다(self, sample_frame):
        src = ReplayDataSource(sample_frame)
        for s in src.iter_steps(datetime(2026, 5, 2), datetime(2026, 5, 3)):
            assert s.history("z1").index.max() <= pd.Timestamp(s.now())


class TestReplayClock:
    def test_step_과_seek(self, sample_frame):
        src = ReplayDataSource(sample_frame, start=datetime(2026, 5, 1, 0, 0))
        assert src.now() == datetime(2026, 5, 1, 0, 0)
        assert src.step() == datetime(2026, 5, 1, 0, 10)
        assert src.step(5) == datetime(2026, 5, 1, 1, 0)
        assert src.seek(datetime(2026, 5, 2, 8, 0)) == datetime(2026, 5, 2, 8, 0)

    def test_격자에_없는_시각은_직전_격자로(self, sample_frame):
        src = ReplayDataSource(sample_frame)
        assert src.seek(datetime(2026, 5, 1, 0, 7)) == datetime(2026, 5, 1, 0, 0)

    def test_끝을_넘어가지_않는다(self, sample_frame):
        src = ReplayDataSource(sample_frame)
        src.step(10_000)
        assert src.at_end
        assert src.now() == sample_frame.index[-1].to_pydatetime()

    def test_시작_이전으로_가지_않는다(self, sample_frame):
        src = ReplayDataSource(sample_frame)
        src.step(-100)
        assert src.now() == sample_frame.index[0].to_pydatetime()

    def test_latest_가_현재_시각의_값을_준다(self, sample_frame):
        src = ReplayDataSource(sample_frame, start=datetime(2026, 5, 1, 12, 30))
        r = src.latest("z1")
        assert r.timestamp == datetime(2026, 5, 1, 12, 30)
        assert r.zone_id == "z1"
        assert r.is_measured

    def test_알수없는_구역은_에러(self, sample_frame):
        src = ReplayDataSource(sample_frame)
        with pytest.raises(KeyError):
            src.history("없는구역")


class TestIngestHelpers:
    def test_로거id_추출은_시리얼을_보존한다(self):
        assert logger_id_of("z6-21068_061225-0906.xlsx") == "z6-21068"
        assert logger_id_of("7구역-z6-21068_061225-0906.xlsx") == "7구역-z6-21068"

    def test_컬럼_부분일치_탐색(self):
        df = pd.DataFrame(columns=["Timestamp", " µmol·m⁻²·s⁻¹ PPFD", " °C Air Temperature"])
        assert find_column(df, "PPFD") == " µmol·m⁻²·s⁻¹ PPFD"
        assert find_column(df, "없는것") is None

    def test_컬럼_탐색_제외어(self):
        df = pd.DataFrame(columns=["Air Temperature", "Soil Temperature"])
        assert find_column(df, "temperature", exclude=("soil",)) == "Air Temperature"

    def test_격자화가_결측_시각을_채운다(self):
        idx = pd.DatetimeIndex(["2026-05-01 00:00", "2026-05-01 00:10", "2026-05-01 00:40"])
        grid, report = build_grid({"a": pd.Series([1.0, 2.0, 3.0], index=idx)}, 10)
        assert len(grid) == 5                      # 00:00 ~ 00:40
        assert grid["a"].isna().sum() == 2         # 00:20, 00:30 이 빈 행으로 삽입됨

    def test_부동소수점_드리프트를_격자로_정규화(self):
        """19:59:59.995 같은 값이 남으면 외부 데이터와 timestamp 매칭이 전부 깨진다."""
        idx = pd.DatetimeIndex(["2026-05-01 00:00:00", "2026-05-01 00:09:59.995"])
        grid, report = build_grid({"a": pd.Series([1.0, 2.0], index=idx)}, 10)
        assert list(grid.index) == [pd.Timestamp("2026-05-01 00:00"),
                                    pd.Timestamp("2026-05-01 00:10")]
        assert report["drift_fixed"] >= 1

    def test_중복_timestamp_는_첫_값을_남긴다(self):
        idx = pd.DatetimeIndex(["2026-05-01 00:00", "2026-05-01 00:00"])
        grid, report = build_grid({"a": pd.Series([1.0, 9.0], index=idx)}, 10)
        assert report["duplicates_dropped"] == 1
        assert grid["a"].iloc[0] == 1.0


class TestSiteFrame:
    def test_slice_는_반개구간(self, sample_frame):
        sub = sample_frame.slice("2026-05-02", "2026-05-03")
        assert sub.index.min() == pd.Timestamp("2026-05-02 00:00")
        assert sub.index.max() == pd.Timestamp("2026-05-02 23:50")

    def test_출처_플래그가_생성된다(self, sample_frame):
        assert set(sample_frame.source["z1"].unique()) <= {"measured", "missing"}
