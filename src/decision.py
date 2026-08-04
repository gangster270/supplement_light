"""P4: 보광 점등 판단 엔진.

★ 이 모듈은 **순수 함수**다. pandas / 파일 IO / Streamlit 에 의존하지 않는다.
   그래야 백테스트(P5)와 실시간 UI(P6)가 문자 그대로 같은 코드를 탄다.
   판단 로직이 UI 쪽에 조금이라도 새어 나가면 백테스트 결과와 화면이 달라진다.

계층 구조 (위가 아래를 덮어쓴다):

  S0 안전 조건   고온 차단 / 센서 이상 (광량 판단보다 먼저)
  L0 실험 제약   NI 스케줄 / 일일 점등 상한 / 광주기 상한 / 허용 시간대
  L1 안정화      최소 점등·소등 유지시간 (채터링 방지)
  L2 DLI 부족분  목표 DLI 를 못 채울 전망이면 점등            ← 주 판단
  L3 순간 임계   맑아야 할 시각인데 어두우면 점등 (저광일 대응)  ← 보조
  L4 경제성      급하지 않으면 저가 시간대로 배치

판단값만이 아니라 **근거**(발동 레이어·각 수치)를 함께 반환한다.
근거 없는 신호는 농가가 신뢰하지 않고, 나중에 왜 그랬는지 검증도 못 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .config import DecisionConfig, LampConfig, NIConfig, SafetyConfig, SiteConfig
from .tariff import TariffLookup
from .timeutil import day_bounds, in_any_window


class Signal(str, Enum):
    """등기구가 있어야 할 상태."""

    ON = "ON"
    OFF = "OFF"


class Action(str, Enum):
    """현재 상태 대비 취해야 할 행동. UI의 3색 신호등이 이걸 그대로 쓴다."""

    TURN_ON = "점등"
    KEEP_ON = "유지(점등 중)"
    TURN_OFF = "소등"
    KEEP_OFF = "유지(소등 중)"

    @property
    def is_change(self) -> bool:
        return self in (Action.TURN_ON, Action.TURN_OFF)


class DisplayStatus(str, Enum):
    """화면에 띄우는 4가지 상태. 신호(ON/OFF)보다 사람이 읽기 쉬운 층위다."""

    RECOMMEND_ON = "보광 권장"
    NOT_NEEDED = "보광 불필요"
    DEFER = "보광 보류"
    CHECK_SENSOR = "센서 확인"


class Layer(str, Enum):
    SENSOR_FAULT = "S0-센서 이상"
    HIGH_TEMP = "S0-고온 차단"
    NI = "L0-NI 스케줄"
    DAILY_CAP = "L0-일일 점등상한"
    PHOTOPERIOD_CAP = "L0-광주기 상한"
    OUTSIDE_WINDOW = "L0-허용 시간대 밖"
    HOLD = "L1-최소 유지시간"
    DLI_URGENT = "L2-DLI 부족(긴급)"
    DLI_SUFFICIENT = "L2-DLI 충족 전망"
    THRESHOLD = "L3-저광 임계"
    TARIFF = "L4-저가 시간대 배치"
    TARIFF_DEFER = "L4-저가 시간대로 연기"
    NATURAL_SUFFICIENT = "자연광 충분"


@dataclass(frozen=True)
class DecisionParams:
    """판단에 필요한 설정 묶음. SiteConfig 에서 build_params() 로 만든다."""

    decision: DecisionConfig
    lamp: LampConfig
    ni: NIConfig
    interval_minutes: int
    safety: SafetyConfig | None = None
    tariff: TariffLookup | None = None

    @property
    def interval_hours(self) -> float:
        return self.interval_minutes / 60.0


@dataclass(frozen=True)
class DecisionState:
    """판단 시점의 상태. 여기 들어오는 값은 모두 '지금까지 관측된 것'이어야 한다."""

    now: datetime
    zone_id: str
    treatment: str
    ppfd_ma: float | None            # 이동평균 PPFD (등기구 기여 포함된 실측)
    dli_today: float                 # 당일 00:00~현재 누적 DLI
    forecast_remaining: float        # 남은 자연광 예상 DLI
    lamp_on: bool                    # 현재 등기구 상태
    minutes_in_state: int            # 현재 상태 유지 시간(분)
    lighting_minutes_today: float    # 당일 누적 점등시간(분)
    clear_sky_ppfd: float = 0.0      # 지금이 맑았다면 나왔을 PPFD (L3 게이트)
    ppfd_now: float | None = None    # 원시 최신값 (표시용)
    ppfd_source: str = "measured"    # 추정값 기반 판단인지 표시
    forecast_low: float | None = None
    forecast_high: float | None = None
    photoperiod_minutes_today: float | None = None
    forecast_note: str = ""
    forecast_confidence: str = "high"
    air_temperature: float | None = None      # 안전 조건용 (없으면 고온 판단 생략)
    sensor_age_minutes: float | None = None   # 마지막 유효 측정 이후 경과 분


@dataclass(frozen=True)
class Decision:
    """판단 결과 + 근거. 그대로 판단 이력(P7)에 기록된다."""

    signal: Signal
    action: Action
    layer: Layer
    reason: str
    deficit_dli: float = 0.0
    hours_needed: float = 0.0
    hours_available: float = 0.0
    slack_hours: float = 0.0
    natural_ppfd_estimate: float | None = None
    planned_ni_dli: float = 0.0
    projected_dli: float = 0.0
    price_now: float | None = None
    price_slot: str = ""
    blocked_by: str | None = None
    warning: str | None = None
    evidence: dict = field(default_factory=dict)

    @property
    def should_be_on(self) -> bool:
        return self.signal is Signal.ON

    @property
    def status(self) -> DisplayStatus:
        """화면 표시용 4상태. ON/OFF 만으로는 '지금은 보류'인지 '필요 없음'인지 구분되지 않는다."""
        if self.layer is Layer.SENSOR_FAULT:
            return DisplayStatus.CHECK_SENSOR
        if self.signal is Signal.ON:
            return DisplayStatus.RECOMMEND_ON
        if self.layer in (Layer.TARIFF_DEFER, Layer.HOLD, Layer.HIGH_TEMP):
            return DisplayStatus.DEFER
        return DisplayStatus.NOT_NEEDED


# =====================================================================
# 보조 계산
# =====================================================================

def build_params(cfg: SiteConfig) -> DecisionParams:
    return DecisionParams(
        decision=cfg.decision,
        lamp=cfg.lamp,
        ni=cfg.ni,
        interval_minutes=cfg.interval_minutes,
        safety=cfg.safety,
        tariff=TariffLookup(cfg.tariff),
    )


def _remaining_slots(now: datetime, params: DecisionParams, ni_applies: bool
                     ) -> tuple[list[datetime], list[datetime]]:
    """지금부터 그날 자정까지의 남은 슬롯을 (재량 슬롯, NI 강제점등 슬롯)로 나눈다.

    NI 슬롯은 어차피 켜지므로 재량 슬롯에서 빼고, 그 기여분은 따로 DLI에 더한다.
    (이걸 안 하면 밤에 어차피 켤 불을 낮에 또 켜자고 판단한다.)
    """
    _, day_end = day_bounds(now)
    step = timedelta(minutes=params.interval_minutes)
    discretionary, ni_slots = [], []
    cur = now
    while cur < day_end:
        if ni_applies and params.ni.window.contains(cur):
            ni_slots.append(cur)
        elif in_any_window(cur, params.decision.allowed_windows):
            discretionary.append(cur)
        cur += step
    return discretionary, ni_slots


def _select_cheapest(slots: list[datetime], hours_needed: float,
                     params: DecisionParams) -> list[datetime]:
    """필요 점등시간만큼을 가장 싼 슬롯부터 고른다. 동일 단가면 이른 시각 우선."""
    if not slots or hours_needed <= 0:
        return []
    n = min(len(slots), math.ceil(hours_needed / params.interval_hours))
    if params.tariff is None:
        return slots[:n]
    ordered = sorted(slots, key=lambda t: (params.tariff.price(t), t))
    return ordered[:n]


def estimate_natural_ppfd(state: DecisionState, params: DecisionParams) -> float | None:
    """실측 이동평균에서 등기구 기여분을 빼 '자연광만'을 추정한다.

    등기구가 켜진 상태의 실측값을 그대로 임계값과 비교하면, 보광 때문에 밝아진 것을
    '자연광이 충분해졌다'고 오판해서 껐다 켰다를 반복한다.
    """
    if state.ppfd_ma is None or (isinstance(state.ppfd_ma, float) and math.isnan(state.ppfd_ma)):
        return None
    value = state.ppfd_ma - (params.lamp.ppfd_contribution if state.lamp_on else 0.0)
    return max(value, 0.0)


def _finalize(state: DecisionState, params: DecisionParams, desired: Signal,
              layer: Layer, reason: str, bypass_hold: bool = False,
              warning: str | None = None, **kw) -> Decision:
    """L1(최소 유지시간)을 적용해 최종 행동을 정한다."""
    d = params.decision
    price_now = params.tariff.price(state.now) if params.tariff else None
    price_slot = params.tariff.slot_name(state.now) if params.tariff else ""
    blocked = None

    wants_change = (desired is Signal.ON) != state.lamp_on
    if wants_change and not bypass_hold:
        hold = d.min_on_minutes if state.lamp_on else d.min_off_minutes
        if state.minutes_in_state < hold:
            remaining = hold - state.minutes_in_state
            blocked = (f"최소 {'점등' if state.lamp_on else '소등'} 유지시간 {hold}분 미달 "
                       f"(앞으로 {remaining}분). 잦은 점·소등을 막기 위해 현재 상태를 유지합니다.")
            desired = Signal.ON if state.lamp_on else Signal.OFF
            layer, reason = Layer.HOLD, blocked
            wants_change = False

    if desired is Signal.ON:
        action = Action.TURN_ON if not state.lamp_on else Action.KEEP_ON
    else:
        action = Action.TURN_OFF if state.lamp_on else Action.KEEP_OFF

    return Decision(signal=desired, action=action, layer=layer, reason=reason,
                    price_now=price_now, price_slot=price_slot, blocked_by=blocked,
                    warning=warning,
                    natural_ppfd_estimate=estimate_natural_ppfd(state, params), **kw)


# =====================================================================
# 판단 엔진
# =====================================================================

def decide(state: DecisionState, params: DecisionParams) -> Decision:
    """한 구역, 한 시점의 점등 판단."""
    d = params.decision
    lamp = params.lamp
    safety = params.safety
    ni_applies = params.ni.applies_to(state.treatment)
    in_ni = ni_applies and params.ni.window.contains(state.now)

    # ---------------- DLI 수지 계산 (모든 분기에서 공통) ----------------
    # ★ 안전·제약 레이어에서 조기 반환하더라도 이 숫자들은 채워서 내보낸다.
    #   안 그러면 화면과 판단 이력에 0 이 찍혀 '충족'처럼 읽힌다 (실제로 발생했던 표시 버그).
    discretionary, ni_slots = _remaining_slots(state.now, params, ni_applies)
    dli_per_hour = lamp.dli_per_hour()
    planned_ni_dli = len(ni_slots) * params.interval_hours * dli_per_hour
    projected = state.dli_today + state.forecast_remaining + planned_ni_dli
    deficit = max(0.0, d.target_dli - projected)
    hours_needed = deficit / dli_per_hour if dli_per_hour > 0 else float("inf")
    hours_available = len(discretionary) * params.interval_hours
    slack = hours_available - hours_needed
    common = dict(deficit_dli=deficit, hours_needed=hours_needed,
                  hours_available=hours_available, slack_hours=slack,
                  planned_ni_dli=planned_ni_dli, projected_dli=projected)

    # ---------------- S0: 안전 조건 (광량 판단보다 먼저) ----------------
    too_hot = safety is not None and safety.is_too_hot(state.air_temperature, state.lamp_on)
    heat_warning = None
    if too_hot:
        heat_warning = (
            f"기온 {state.air_temperature:.1f}℃ 로 고온 한계"
            f"({safety.max_air_temperature:.1f}℃)를 넘었습니다. 등기구는 발열원이므로 "
            f"고온 스트레스를 가중시킵니다.")
        # NI 는 실험 처리 조건이라 기본적으로 고온에도 유지한다 (설정으로 변경 가능).
        if not in_ni or safety.override_ni_on_high_temp:
            note = (" NI 구간이지만 안전 우선 설정(override_ni_on_high_temp: true)에 따라 "
                    "중단합니다 — 실험 처리 조건이 깨지므로 연구자 확인이 필요합니다."
                    if in_ni else "")
            return _finalize(
                state, params, Signal.OFF, Layer.HIGH_TEMP,
                heat_warning + " 보광을 중단합니다." + note,
                bypass_hold=True, warning=heat_warning,
                evidence={"air_temperature": state.air_temperature,
                          "max_air_temperature": safety.max_air_temperature,
                          "overrode_ni": in_ni, "target_dli": d.target_dli}, **common)

    if (safety is not None and state.sensor_age_minutes is not None
            and state.sensor_age_minutes > safety.max_sensor_age_minutes and not in_ni):
        # 죽은 센서 값으로 계속 판단하면 틀린 근거로 보광을 권고하게 된다.
        # 임의로 켜거나 끄지 않고 현재 상태를 유지한 채 사람에게 넘긴다.
        return _finalize(
            state, params, Signal.ON if state.lamp_on else Signal.OFF, Layer.SENSOR_FAULT,
            f"마지막 유효 측정이 {state.sensor_age_minutes:.0f}분 전입니다 "
            f"(허용 {safety.max_sensor_age_minutes:.0f}분). 센서·통신 상태를 확인하세요. "
            f"확인 전까지 현재 상태({'점등' if state.lamp_on else '소등'})를 유지합니다.",
            bypass_hold=True, warning="센서 데이터가 갱신되지 않고 있습니다.",
            evidence={"sensor_age_minutes": state.sensor_age_minutes,
                      "ppfd_source": state.ppfd_source, "target_dli": d.target_dli}, **common)

    # ---------------- L0: 실험 제약 (최소유지시간도 무시) ----------------
    if in_ni:
        return _finalize(
            state, params, Signal.ON, Layer.NI,
            f"NI(야간중단) 구간 {params.ni.window} 입니다. 실험 처리 조건이므로 "
            f"광량과 무관하게 점등을 유지합니다."
            + (" ※ 다만 현재 고온 상태입니다." if too_hot else ""),
            bypass_hold=True, warning=heat_warning,
            evidence={"ni_window": str(params.ni.window), "treatment": state.treatment,
                      "high_temp": too_hot, "target_dli": d.target_dli}, **common)

    max_daily_minutes = d.max_daily_lighting_hours * 60
    if state.lighting_minutes_today >= max_daily_minutes:
        return _finalize(
            state, params, Signal.OFF, Layer.DAILY_CAP,
            f"오늘 누적 점등 {state.lighting_minutes_today / 60:.1f}시간으로 상한"
            f"({d.max_daily_lighting_hours:.1f}시간)에 도달했습니다. 추가 점등하지 않습니다.",
            bypass_hold=True,
            evidence={"lighting_hours_today": state.lighting_minutes_today / 60,
                      "target_dli": d.target_dli}, **common)

    if (state.photoperiod_minutes_today is not None
            and state.photoperiod_minutes_today >= d.max_photoperiod_hours * 60):
        return _finalize(
            state, params, Signal.OFF, Layer.PHOTOPERIOD_CAP,
            f"오늘 광주기가 상한({d.max_photoperiod_hours:.1f}시간)에 도달했습니다.",
            bypass_hold=True,
            evidence={"photoperiod_hours": state.photoperiod_minutes_today / 60,
                      "target_dli": d.target_dli}, **common)

    if not in_any_window(state.now, d.allowed_windows):
        windows = ", ".join(str(w) for w in d.allowed_windows) or "(설정 없음)"
        return _finalize(
            state, params, Signal.OFF, Layer.OUTSIDE_WINDOW,
            f"보광 허용 시간대({windows}) 밖입니다.",
            bypass_hold=True,
            evidence={"allowed_windows": windows, "target_dli": d.target_dli}, **common)

    natural = estimate_natural_ppfd(state, params)
    evidence = {
        "dli_today": state.dli_today,
        "forecast_remaining": state.forecast_remaining,
        "forecast_low": state.forecast_low,
        "forecast_high": state.forecast_high,
        "forecast_note": state.forecast_note,
        "forecast_confidence": state.forecast_confidence,
        "target_dli": d.target_dli,
        "ppfd_ma": state.ppfd_ma,
        "natural_ppfd": natural,
        "clear_sky_ppfd": state.clear_sky_ppfd,
        "ppfd_source": state.ppfd_source,
        "n_discretionary_slots": len(discretionary),
        "n_ni_slots": len(ni_slots),
    }

    # 자연광이 지금 충분하면 켜지 않는다 (등기구 기여분을 뺀 값으로 판단).
    natural_sufficient = natural is not None and natural > d.off_threshold

    # ---------------- L3: 저광 임계 (보조) ----------------
    # 맑았다면 밝아야 할 시각인데 실제로 어두운 날 = 저광일. 밤에는 clear_sky_ppfd 가
    # 0에 가까워 자동으로 발동하지 않는다 (밤에 임계값만 보면 항상 점등이 된다).
    dark_day = (natural is not None
                and state.clear_sky_ppfd > d.on_threshold
                and natural < d.on_threshold)

    if deficit <= 0:
        return _finalize(
            state, params, Signal.OFF, Layer.DLI_SUFFICIENT,
            f"오늘 목표 DLI {d.target_dli:.1f} 도달 전망입니다 "
            f"(현재 {state.dli_today:.1f} + 자연광 예상 {state.forecast_remaining:.1f}"
            + (f" + NI 기여 {planned_ni_dli:.1f}" if planned_ni_dli > 0 else "")
            + f" = {projected:.1f} mol). 보광이 필요하지 않습니다.",
            evidence=evidence, **common)

    if natural_sufficient:
        return _finalize(
            state, params, Signal.OFF, Layer.NATURAL_SUFFICIENT,
            f"부족분 {deficit:.1f} mol 이 남아 있으나 현재 자연광이 "
            f"{natural:.0f} µmol 로 충분합니다(소등 임계 {d.off_threshold:.0f}). "
            f"지금 켜는 것은 전력 낭비이므로 어두워지면 다시 판단합니다.",
            evidence=evidence, **common)

    urgent = slack <= d.urgency_margin_hours

    if urgent:
        if hours_needed > hours_available:
            # 켜도 목표에 못 미치는 상황. "켜면 채워진다"고 읽히면 안 된다 —
            # 연구자가 목표 DLI나 등기구 용량을 재검토해야 하는 신호다.
            shortfall = (hours_needed - hours_available) * dli_per_hour
            reason = (f"오늘은 목표 DLI 달성이 불가능합니다. 부족분 {deficit:.1f} mol 을 "
                      f"채우려면 {hours_needed:.1f}시간이 필요한데 남은 가용시간은 "
                      f"{hours_available:.1f}시간뿐입니다 (약 {shortfall:.1f} mol 미달). "
                      f"가능한 만큼 채우기 위해 점등합니다.")
        else:
            reason = (f"지금 켜지 않으면 오늘 목표를 채우지 못합니다. 부족분 {deficit:.1f} mol "
                      f"→ 필요 점등 {hours_needed:.1f}시간, 남은 가용시간 "
                      f"{hours_available:.1f}시간 (여유 {slack:.1f}시간).")
        return _finalize(state, params, Signal.ON, Layer.DLI_URGENT, reason,
                         evidence=evidence, **common)

    # ---------------- L4: 급하지 않으면 저가 시간대로 ----------------
    cheapest = _select_cheapest(discretionary, hours_needed, params)
    selected_now = bool(cheapest) and state.now in set(cheapest)
    price_now = params.tariff.price(state.now) if params.tariff else None
    evidence["n_selected_slots"] = len(cheapest)
    if cheapest and params.tariff is not None:
        evidence["selected_price_max"] = max(params.tariff.price(t) for t in cheapest)

    if selected_now:
        return _finalize(
            state, params, Signal.ON, Layer.TARIFF,
            f"부족분 {deficit:.1f} mol (필요 {hours_needed:.1f}시간). 지금은 남은 시간 중 "
            f"단가가 낮은 구간({params.tariff.slot_name(state.now) if params.tariff else ''}"
            f"{f', {price_now:.0f}원/kWh' if price_now is not None else ''})이라 "
            f"지금 점등하는 것이 유리합니다.",
            evidence=evidence, **common)

    if dark_day:
        return _finalize(
            state, params, Signal.ON, Layer.THRESHOLD,
            f"맑았다면 {state.clear_sky_ppfd:.0f} µmol 이어야 할 시각인데 실제 "
            f"{natural:.0f} µmol 로 점등 임계({d.on_threshold:.0f}) 미만인 저광 상태입니다. "
            f"부족분 {deficit:.1f} mol 을 감안해 점등합니다.",
            evidence=evidence, **common)

    next_slot = min(cheapest) if cheapest else None
    return _finalize(
        state, params, Signal.OFF, Layer.TARIFF_DEFER,
        f"부족분 {deficit:.1f} mol (필요 {hours_needed:.1f}시간)이 있으나 여유 "
        f"{slack:.1f}시간이 남아 있어, 단가가 더 낮은 시간대"
        + (f"({next_slot:%H:%M} 부터)" if next_slot else "")
        + "로 점등을 미룹니다.",
        evidence=evidence, **common)
