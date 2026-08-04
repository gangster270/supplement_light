"""제어 명령 미리보기.

★ 이 모듈은 **아무것도 제어하지 않는다.** 나중에 PLC·복합환경제어기와 연결한다면
   어떤 명령이 나갈지를 JSON 으로 보여주기만 한다. 시제품 단계에서 실제 하드웨어를
   건드리면 오작동 시 실험 자체를 망친다.

실제 연동 시 확인할 것:
  - 제어기 프로토콜(Modbus TCP / BACnet / 제조사 REST)과 레지스터·엔드포인트 매핑
  - 명령 실패·타임아웃 시 폴백 (마지막 상태 유지가 기본)
  - 워치독: 서비스가 죽으면 제어기가 스스로 안전 상태로 돌아가야 한다
  - 수동 조작 우선권 (현장에서 사람이 끈 것을 서비스가 다시 켜면 안 된다)
"""

from __future__ import annotations

import json

from .decision import Decision, DecisionState, Signal

SCHEMA_VERSION = "0.1-preview"


def build_command(state: DecisionState, decision: Decision, zone_name: str = "",
                  greenhouse: str = "", dry_run: bool = True) -> dict:
    """판단 결과를 제어 명령 형태(JSON 직렬화 가능한 dict)로 변환한다."""
    return {
        "schema_version": SCHEMA_VERSION,
        "dry_run": dry_run,          # 시제품에서는 항상 True
        "issued_at": state.now.isoformat(),
        "target": {
            "greenhouse": greenhouse,
            "zone_id": state.zone_id,
            "zone_name": zone_name,
            "treatment": state.treatment,
            "device": "supplemental_light",
        },
        "command": {
            "action": "ON" if decision.signal is Signal.ON else "OFF",
            "current_state": "ON" if state.lamp_on else "OFF",
            "is_change": decision.action.is_change,
            "hold_seconds": None,    # 제어기 측 최소 유지시간은 서비스가 이미 반영함
        },
        "rationale": {
            "status": decision.status.value,
            "layer": decision.layer.value,
            "reason": decision.reason,
            "warning": decision.warning,
            "blocked_by": decision.blocked_by,
        },
        "measurements": {
            "ppfd_now": state.ppfd_now,
            "ppfd_moving_average": state.ppfd_ma,
            "natural_ppfd_estimate": decision.natural_ppfd_estimate,
            "air_temperature": state.air_temperature,
            "ppfd_source": state.ppfd_source,
            "sensor_age_minutes": state.sensor_age_minutes,
        },
        "dli": {
            "today": round(state.dli_today, 3),
            "forecast_remaining": round(state.forecast_remaining, 3),
            "forecast_low": None if state.forecast_low is None else round(state.forecast_low, 3),
            "forecast_high": None if state.forecast_high is None else round(state.forecast_high, 3),
            "planned_ni": round(decision.planned_ni_dli, 3),
            "projected": round(decision.projected_dli, 3),
            "deficit": round(decision.deficit_dli, 3),
            "hours_needed": round(decision.hours_needed, 2),
        },
    }


def to_json(command: dict) -> str:
    return json.dumps(command, ensure_ascii=False, indent=2)
