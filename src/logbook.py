"""판단 이력 기록.

매 스텝의 판단을 근거 수치까지 함께 남긴다. 나중에 "왜 그때 켰냐"를 검증할 수
없으면 연구 도구로 쓸 수 없다.

메모리에 쌓고 필요할 때 DataFrame/CSV 로 내보낸다. 시제품 단계에서는 DB를 쓰지 않는다
(운영 부담만 늘고 시연에는 도움이 안 된다).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .decision import Decision, DecisionState

COLUMNS = [
    "시각", "구역", "처리", "상태", "신호", "행동", "발동레이어",
    "PPFD_실측", "PPFD_이동평균", "자연광_추정", "기온",
    "누적DLI", "잔여자연광_예상", "예상_하한", "예상_상한",
    "NI_기여", "예상최종DLI", "목표DLI", "부족DLI",
    "필요점등시간", "가용시간", "여유시간",
    "단가", "부하구간", "값출처", "경고", "보류사유", "근거",
]


@dataclass
class LogBook:
    """판단 이력 누적기."""

    records: list[dict] = field(default_factory=list)

    def append(self, state: DecisionState, decision: Decision,
               zone_name: str = "") -> dict:
        ev = decision.evidence
        row = {
            "시각": state.now,
            "구역": zone_name or state.zone_id,
            "처리": state.treatment,
            "상태": decision.status.value,
            "신호": decision.signal.value,
            "행동": decision.action.value,
            "발동레이어": decision.layer.value,
            "PPFD_실측": state.ppfd_now,
            "PPFD_이동평균": state.ppfd_ma,
            "자연광_추정": decision.natural_ppfd_estimate,
            "기온": state.air_temperature,
            "누적DLI": state.dli_today,
            "잔여자연광_예상": state.forecast_remaining,
            "예상_하한": state.forecast_low,
            "예상_상한": state.forecast_high,
            "NI_기여": decision.planned_ni_dli,
            "예상최종DLI": decision.projected_dli,
            "목표DLI": ev.get("target_dli"),
            "부족DLI": decision.deficit_dli,
            "필요점등시간": decision.hours_needed,
            "가용시간": decision.hours_available,
            "여유시간": decision.slack_hours,
            "단가": decision.price_now,
            "부하구간": decision.price_slot,
            "값출처": state.ppfd_source,
            "경고": decision.warning,
            "보류사유": decision.blocked_by,
            "근거": decision.reason,
        }
        self.records.append(row)
        return row

    def __len__(self) -> int:
        return len(self.records)

    def clear(self) -> None:
        self.records.clear()

    def to_frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=COLUMNS)
        return pd.DataFrame(self.records)[COLUMNS]

    def to_csv_bytes(self) -> bytes:
        """엑셀에서 한글이 깨지지 않도록 BOM 을 붙인다."""
        return self.to_frame().to_csv(index=False).encode("utf-8-sig")

    def summary(self, interval_minutes: int, power_kw: float = 0.0) -> dict:
        """이력 요약 — 상태 분포, 점등시간, 전력량."""
        df = self.to_frame()
        if df.empty:
            return {"steps": 0, "점등시간": 0.0, "kWh": 0.0, "상태분포": {},
                    "요금": 0.0}
        on = df[df["신호"] == "ON"]
        hours = len(on) * interval_minutes / 60.0
        kwh = hours * power_kw
        # 요금은 각 스텝의 단가로 계산한다 (계시별 단가가 시간대마다 다르므로 평균 곱셈은 틀린다)
        won = float((on["단가"].fillna(0) * power_kw * interval_minutes / 60.0).sum())
        return {
            "steps": len(df),
            "점등시간": hours,
            "kWh": kwh,
            "요금": won,
            "상태분포": df["상태"].value_counts().to_dict(),
            "레이어분포": df["발동레이어"].value_counts().to_dict(),
        }
