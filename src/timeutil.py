"""시각/시간창(window) 계산 유틸.

NI 구간(18:30~02:30)처럼 자정을 넘어가는(wrap) 시간창을 다뤄야 하므로,
"시작 > 종료면 자정을 넘는 구간"이라는 규칙을 한 곳에서만 구현한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta


def parse_time(text: str) -> time:
    """'18:30' / '18:30:00' 형식을 time 으로."""
    parts = [int(p) for p in str(text).strip().split(":")]
    while len(parts) < 3:
        parts.append(0)
    return time(parts[0], parts[1], parts[2])


@dataclass(frozen=True)
class TimeWindow:
    """하루 안의 시간창. start > end 이면 자정을 넘어가는 구간으로 해석한다."""

    start: time
    end: time
    include_start: bool = True
    include_end: bool = False
    label: str = ""

    @classmethod
    def parse(cls, text: str, include_start: bool = True, include_end: bool = False,
              label: str = "") -> "TimeWindow":
        """'04:00-08:00' 형식 파싱."""
        left, right = str(text).split("-")
        return cls(parse_time(left), parse_time(right), include_start, include_end, label)

    @property
    def wraps_midnight(self) -> bool:
        return self.start > self.end

    def contains(self, moment: datetime | time) -> bool:
        t = moment.time() if isinstance(moment, datetime) else moment
        after_start = t >= self.start if self.include_start else t > self.start
        before_end = t <= self.end if self.include_end else t < self.end
        if self.wraps_midnight:
            # 자정을 넘는 구간은 [start, 24:00) 또는 [00:00, end] 중 하나에 들어가면 참
            return after_start or before_end
        return after_start and before_end

    def __str__(self) -> str:
        return f"{self.start:%H:%M}-{self.end:%H:%M}"


def in_any_window(moment: datetime | time, windows: list[TimeWindow]) -> bool:
    return any(w.contains(moment) for w in windows)


def day_bounds(moment: datetime) -> tuple[datetime, datetime]:
    """해당 달력일의 [00:00, 다음날 00:00) 경계.

    DLI는 기존 QC 파이프라인과 동일하게 '달력일' 기준으로 집계한다
    (NI가 자정을 넘더라도 DLI 집계 기준은 바꾸지 않는다 — 기존 산출물과 대조 가능해야 하므로).
    """
    start = datetime.combine(moment.date(), time(0, 0))
    return start, start + timedelta(days=1)


def minute_of_day(moment: datetime) -> int:
    return moment.hour * 60 + moment.minute


def slot_range(start: datetime, end: datetime, interval_minutes: int) -> list[datetime]:
    """[start, end) 구간을 interval 간격 슬롯 시작시각 리스트로."""
    out, cur, step = [], start, timedelta(minutes=interval_minutes)
    while cur < end:
        out.append(cur)
        cur += step
    return out


def circular_doy_distance(a: int, b: int, year_length: int = 366) -> int:
    """연도 경계를 넘는 DOY 거리 (12/31 과 1/1 은 1일 차이)."""
    d = abs(a - b)
    return min(d, year_length - d)


def day_of_year(d: date | datetime) -> int:
    return d.timetuple().tm_yday
