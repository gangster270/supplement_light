"""ZENTRA Cloud API 어댑터 (스텁).

P1 범위에서는 자리만 만들어 둔다. 리플레이와 동일한 DataSource 인터페이스를 구현하므로,
토큰이 확보되면 이 파일의 `_fetch()` 만 채우면 판단 엔진·UI 수정 없이 실시간으로 전환된다.

구현 시 확인할 것:
  - 인증: ZENTRA Cloud API 토큰 (Authorization: Token <key>)
  - 엔드포인트: /api/v4/get_readings/  (device_sn, start_date, end_date)
  - 요청 제한(rate limit): 기기당 1분 1회 수준. 폴링 주기를 측정간격(10분)에 맞추고
    실패 시 지수 백오프로 재시도할 것.
  - 반환 단위가 site.yaml 의 가정과 같은지 반드시 대조 (Processed vs Raw 문제와 동일).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .base import DataSource, Reading

NOT_IMPLEMENTED_MSG = (
    "ZENTRA Cloud 실시간 연동은 아직 구현되지 않았습니다. "
    "현재는 ReplayDataSource(과거 데이터 리플레이)를 사용하세요. "
    "연동하려면 src/datasource/zentra.py 의 _fetch() 를 구현하고 API 토큰을 설정하세요."
)


class ZentraDataSource(DataSource):
    def __init__(self, token: str, device_map: dict[str, str], interval_minutes: int = 10):
        """device_map: {zone_id: device_serial}"""
        self.token = token
        self.device_map = dict(device_map)
        self.interval_minutes = interval_minutes

    @property
    def zone_ids(self) -> list[str]:
        return list(self.device_map)

    def now(self) -> datetime:
        return datetime.now()

    def _fetch(self, device_serial: str, start: datetime, end: datetime) -> pd.Series:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def latest(self, zone_id: str) -> Reading | None:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def history(self, zone_id: str, start: datetime | None = None,
                end: datetime | None = None) -> pd.Series:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def external_history(self, start: datetime | None = None,
                         end: datetime | None = None) -> pd.Series:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)
