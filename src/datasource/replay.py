"""과거 데이터 리플레이 데이터소스.

SiteFrame(과거 전체 기간)을 들고 있으면서 "가상 현재시각" 커서를 움직인다.
커서 이후의 데이터는 존재하더라도 절대 내보내지 않는다 (no lookahead).

두 가지 사용 방식:
  1) 수동 스텝: step() / seek() — 백테스트와 UI의 스텝 버튼
  2) 배속 재생: start_playback(speed) 후 sync() — UI의 재생 버튼
     speed=60 이면 실제 1초에 가상 60초가 흐른다.
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta

import pandas as pd

from ..ingest import SiteFrame
from .base import DataSource, Reading


class ReplayDataSource(DataSource):
    def __init__(self, frame: SiteFrame, start: datetime | None = None, speed: float = 60.0):
        if len(frame.index) == 0:
            raise ValueError("리플레이할 데이터가 비어 있습니다.")
        self._frame = frame
        self.interval_minutes = frame.interval_minutes
        self._index = frame.index
        self._speed = float(speed)
        self._playing = False
        self._wall_anchor: float | None = None
        self._virtual_anchor: datetime | None = None
        self._cursor = 0
        self.seek(start if start is not None else self._index[0])

    # --- 범위 ---------------------------------------------------------
    @property
    def zone_ids(self) -> list[str]:
        return self._frame.zone_ids

    @property
    def available_range(self) -> tuple[datetime, datetime]:
        return self._index[0].to_pydatetime(), self._index[-1].to_pydatetime()

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def at_end(self) -> bool:
        return self._cursor >= len(self._index) - 1

    @property
    def speed(self) -> float:
        return self._speed

    # --- 시계 ---------------------------------------------------------
    def now(self) -> datetime:
        return self._index[self._cursor].to_pydatetime()

    def seek(self, moment: datetime) -> datetime:
        """가장 가까운(이하) 격자 시각으로 커서를 옮긴다."""
        ts = pd.Timestamp(moment)
        pos = self._index.searchsorted(ts, side="right") - 1
        self._cursor = int(min(max(pos, 0), len(self._index) - 1))
        if self._playing:
            self._anchor_now()
        return self.now()

    def step(self, steps: int = 1) -> datetime:
        self._cursor = int(min(max(self._cursor + steps, 0), len(self._index) - 1))
        if self._playing:
            self._anchor_now()
        return self.now()

    def reset(self) -> datetime:
        return self.seek(self._index[0])

    # --- 배속 재생 -----------------------------------------------------
    def _anchor_now(self) -> None:
        self._wall_anchor = _time.monotonic()
        self._virtual_anchor = self.now()

    def start_playback(self, speed: float | None = None) -> None:
        if speed is not None:
            self._speed = float(speed)
        self._playing = True
        self._anchor_now()

    def pause_playback(self) -> None:
        self.sync()
        self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing

    def sync(self) -> datetime:
        """재생 중이면 실제 경과시간 × speed 만큼 가상 시각을 진행시킨다."""
        if not self._playing or self._wall_anchor is None or self._virtual_anchor is None:
            return self.now()
        elapsed = _time.monotonic() - self._wall_anchor
        target = self._virtual_anchor + timedelta(seconds=elapsed * self._speed)
        if target >= self._index[-1]:
            self._cursor = len(self._index) - 1
            self._playing = False
            return self.now()
        pos = self._index.searchsorted(pd.Timestamp(target), side="right") - 1
        self._cursor = int(min(max(pos, 0), len(self._index) - 1))
        return self.now()

    # --- 데이터 조회 (미래 차단) ----------------------------------------
    def _clip_end(self, end: datetime | None) -> pd.Timestamp:
        cur = self._index[self._cursor]
        return min(pd.Timestamp(end), cur) if end is not None else cur

    def _window_mask(self, start: datetime | None, end: datetime | None):
        end_ts = self._clip_end(end)
        mask = self._index <= end_ts
        if start is not None:
            mask &= self._index >= pd.Timestamp(start)
        return mask

    def history(self, zone_id: str, start: datetime | None = None,
                end: datetime | None = None) -> pd.Series:
        if zone_id not in self._frame.ppfd.columns:
            raise KeyError(f"알 수 없는 구역: {zone_id!r}")
        return self._frame.ppfd.loc[self._window_mask(start, end), zone_id]

    def external_history(self, start: datetime | None = None,
                         end: datetime | None = None) -> pd.Series:
        return self._frame.external.loc[self._window_mask(start, end)]

    def source_history(self, zone_id: str, start: datetime | None = None,
                       end: datetime | None = None) -> pd.Series:
        """각 값의 출처 플래그 시계열 (추정값 기반 판단인지 UI에 표시하기 위함)."""
        return self._frame.source.loc[self._window_mask(start, end), zone_id]

    def temperature_history(self, zone_id: str, start: datetime | None = None,
                            end: datetime | None = None) -> pd.Series:
        if not self._frame.has_temperature or zone_id not in self._frame.temperature.columns:
            return pd.Series(dtype=float)
        return self._frame.temperature.loc[self._window_mask(start, end), zone_id]

    def sensor_age_minutes(self, zone_id: str) -> float | None:
        """마지막 실측(measured) 이후 경과 분. 실측이 하나도 없으면 None."""
        src = self.source_history(zone_id)
        measured = src[src == "measured"]
        if measured.empty:
            return None
        delta = pd.Timestamp(self.now()) - measured.index[-1]
        return float(delta.total_seconds() / 60.0)

    def latest(self, zone_id: str) -> Reading | None:
        ts = self._index[self._cursor]
        value = self._frame.ppfd.at[ts, zone_id]
        ext = self._frame.external.at[ts] if ts in self._frame.external.index else None
        temp = None
        if self._frame.has_temperature and zone_id in self._frame.temperature.columns:
            raw = self._frame.temperature.at[ts, zone_id]
            temp = None if pd.isna(raw) else float(raw)
        return Reading(
            timestamp=ts.to_pydatetime(),
            zone_id=zone_id,
            ppfd=None if pd.isna(value) else float(value),
            source=str(self._frame.source.at[ts, zone_id]),
            external_ppfd=None if ext is None or pd.isna(ext) else float(ext),
            air_temperature=temp,
        )

    # --- 백테스트용 ----------------------------------------------------
    def iter_steps(self, start: datetime | None = None, end: datetime | None = None):
        """[start, end] 구간을 한 스텝씩 진행하며 (자기 자신)을 내준다.

        백테스트가 실시간 루프와 정확히 같은 경로를 타도록 하기 위한 것.
        루프 안에서 self 를 통해 조회하면 자동으로 no-lookahead 가 지켜진다.
        """
        s = pd.Timestamp(start) if start is not None else self._index[0]
        e = pd.Timestamp(end) if end is not None else self._index[-1]
        positions = range(int(self._index.searchsorted(s, side="left")),
                          int(self._index.searchsorted(e, side="right")))
        for pos in positions:
            self._cursor = pos
            yield self

    def full_frame(self) -> SiteFrame:
        """리플레이 대상 전체 프레임 (프로파일 학습 등 '사전 준비' 용도 전용).

        판단 루프 안에서는 절대 쓰지 말 것 — 미래를 보게 된다.
        """
        return self._frame
