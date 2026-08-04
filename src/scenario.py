"""데모용 자연광 시나리오 생성.

실데이터나 업로드 파일이 없어도 시연할 수 있어야 한다. 맑음·흐림·구름 변화 등
전형적인 하루를 만들어 리플레이에 그대로 넣는다.

★ 여기서 만드는 값은 **시연용 모형**이지 관측값이 아니다. 화면에도 그렇게 표시한다.
   물리 구조(일장·태양고도·시간대별 투과율)는 실제와 같은 형태를 따르되,
   구름은 확률 모형이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from .config import SiteConfig
from .ingest import SiteFrame


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    description: str
    cloud_level: float        # 하루 평균 청천지수 (1.0 = 완전 맑음)
    cloud_volatility: float   # 하루 안의 변동 폭
    afternoon_shift: float    # 오후로 갈수록 밝아짐(>1) / 흐려짐(<1)
    peak_temperature: float   # 낮 최고 기온 (℃)


SCENARIOS: dict[str, Scenario] = {
    "clear": Scenario("clear", "맑음",
                      "종일 맑아 자연광만으로 목표 DLI 를 채우는 날",
                      cloud_level=0.95, cloud_volatility=0.05,
                      afternoon_shift=1.0, peak_temperature=26.0),
    "variable": Scenario("variable", "구름 변화",
                         "구름이 오갔다 하는 날 — 채터링 방지 로직이 드러난다",
                         cloud_level=0.62, cloud_volatility=0.38,
                         afternoon_shift=1.0, peak_temperature=24.0),
    "overcast": Scenario("overcast", "흐림",
                         "종일 흐려 보광이 필요한 날",
                         cloud_level=0.30, cloud_volatility=0.10,
                         afternoon_shift=1.0, peak_temperature=21.0),
    "clearing": Scenario("clearing", "오전 흐림 → 오후 갬",
                         "예측이 보수적으로 빗나가는 날 (과잉 점등 위험)",
                         cloud_level=0.45, cloud_volatility=0.12,
                         afternoon_shift=2.0, peak_temperature=23.0),
    "clouding": Scenario("clouding", "오전 맑음 → 오후 흐림",
                         "예측이 낙관적으로 빗나가는 날 (보광 놓칠 위험)",
                         cloud_level=0.70, cloud_volatility=0.12,
                         afternoon_shift=0.30, peak_temperature=22.0),
    "heatwave": Scenario("heatwave", "고온",
                         "맑지만 기온이 한계를 넘어 안전 조건이 발동하는 날",
                         cloud_level=0.92, cloud_volatility=0.06,
                         afternoon_shift=1.0, peak_temperature=35.0),
}

DEFAULT_SCENARIO = "overcast"


# ---------------------------------------------------------------------
# 물리 모형 (tools/make_sample_data.py 와 공유)
# ---------------------------------------------------------------------

def daylength_hours(doy: int) -> float:
    """북위 약 37도의 대략적인 일장."""
    return 12.2 + 2.6 * math.sin(2 * math.pi * (doy - 80) / 365.0)


def clear_sky_external(ts: pd.Timestamp) -> float:
    """맑은 날의 외부 PPFD (µmol·m⁻²·s⁻¹)."""
    doy = ts.dayofyear
    dl = daylength_hours(doy)
    solar_noon = 12.5
    sunrise, sunset = solar_noon - dl / 2, solar_noon + dl / 2
    hour = ts.hour + ts.minute / 60.0
    if not (sunrise < hour < sunset):
        return 0.0
    peak = 1500 + 450 * math.sin(2 * math.pi * (doy - 80) / 365.0)
    return max(0.0, peak * math.sin(math.pi * (hour - sunrise) / dl))


def transmittance(ts: pd.Timestamp, gh_factor: float = 1.0) -> float:
    """시간대별 투과율. 저고도(아침·저녁)에서 골조 차폐로 크게 낮아진다."""
    doy = ts.dayofyear
    dl = daylength_hours(doy)
    sunrise = 12.5 - dl / 2
    hour = ts.hour + ts.minute / 60.0
    phase = (hour - sunrise) / dl
    if not (0 < phase < 1):
        return 0.15 * gh_factor
    return (0.14 + 0.36 * math.sin(math.pi * phase)) * gh_factor


# ---------------------------------------------------------------------
# 시나리오 프레임 생성
# ---------------------------------------------------------------------

def build_scenario_frame(cfg: SiteConfig, scenario_key: str, day: date | datetime,
                         days: int = 1, seed: int = 7) -> SiteFrame:
    """지정 시나리오로 하루(또는 며칠)치 SiteFrame 을 만든다.

    구역별 편차는 site.yaml 의 구역 수만큼 만들되, 온실별로 투과율을 조금 다르게 준다
    (실제로 온실마다 피복재가 달라 투과율이 다르기 때문).
    """
    sc = SCENARIOS.get(scenario_key, SCENARIOS[DEFAULT_SCENARIO])
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(day).normalize()
    index = pd.date_range(start, start + pd.Timedelta(days=days),
                          freq=f"{cfg.interval_minutes}min", inclusive="left")

    clear = np.array([clear_sky_external(ts) for ts in index])

    # 하루 안의 흐림 정도: 기본 수준 × 오후 이동 × 부드러운 잡음
    minutes = np.array([ts.hour * 60 + ts.minute for ts in index], dtype=float)
    progress = np.clip((minutes - 6 * 60) / (12 * 60), 0, 1)     # 06시→18시를 0→1
    shift = sc.afternoon_shift ** progress
    noise = rng.normal(0, 1, len(index))
    smooth = np.convolve(noise, np.ones(9) / 9, mode="same")
    cloud = np.clip(sc.cloud_level * shift * (1 + sc.cloud_volatility * smooth * 3), 0.03, 1.15)

    external = np.clip(clear * cloud + rng.normal(0, 3, len(index)), 0, None)
    external[clear <= 0] = 0.0

    # 기온: 일출 후 상승해 14시경 최고, 야간 최저
    temp_phase = np.sin(np.pi * np.clip((minutes - 5 * 60) / (14 * 60), 0, 1))
    night_low = sc.peak_temperature - 12.0
    temps = night_low + (sc.peak_temperature - night_low) * temp_phase
    temps += rng.normal(0, 0.4, len(index))

    ppfd_cols, temp_cols = {}, {}
    gh_factors = {gh.id: 1.0 - 0.14 * i for i, gh in enumerate(cfg.greenhouses)}
    for gh in cfg.greenhouses:
        for j, zone in enumerate(gh.zones):
            tau = np.array([transmittance(ts, gh_factors[gh.id]) for ts in index])
            sib = 1.0 - 0.09 * j          # 형제 구역 부분 차폐
            inside = external * tau * sib
            if cfg.ni.applies_to(zone.treatment):
                ni_mask = np.array([cfg.ni.window.contains(ts.to_pydatetime()) for ts in index])
                inside = inside + ni_mask * cfg.lamp.ppfd_contribution
            ppfd_cols[zone.id] = np.clip(inside + rng.normal(0, 2.0, len(index)), 0, None)
            temp_cols[zone.id] = temps

    return SiteFrame.from_wide(
        pd.DataFrame(ppfd_cols, index=index),
        pd.Series(external, index=index),
        cfg.interval_minutes,
        report={"synthetic": True, "scenario": sc.key, "scenario_name": sc.name},
        temperature=pd.DataFrame(temp_cols, index=index),
    )


def build_training_frame(cfg: SiteConfig, end_day: date | datetime, days: int = 400,
                         seed: int = 11) -> SiteFrame:
    """청천 프로파일 학습용 과거 이력을 만든다.

    예측기는 1년 이상의 이력이 있어야 계절 외삽을 하지 않는다(README 참고).
    데모에서도 그 조건을 맞춰야 예측 정확도가 실제 운영과 비슷하게 나온다.
    """
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end_day).normalize()
    start = end - pd.Timedelta(days=days)
    index = pd.date_range(start, end, freq=f"{cfg.interval_minutes}min", inclusive="left")

    clear = np.array([clear_sky_external(ts) for ts in index])
    # 날짜별 청천지수 AR(1) — 흐린 날은 이어지는 경향
    dates = pd.Series(index.date, index=index)
    level, daily = 0.75, {}
    for d in sorted(dates.unique()):
        level = 0.72 * level + 0.28 * rng.beta(4.5, 2.0)
        daily[d] = float(np.clip(level, 0.08, 1.0))
    base = dates.map(daily).to_numpy(dtype=float)
    smooth = np.convolve(rng.normal(0, 1, len(index)), np.ones(9) / 9, mode="same")
    cloud = np.clip(base * np.clip(1 + 0.35 * smooth, 0.25, 1.35), 0.02, 1.15)
    external = np.clip(clear * cloud, 0, None)
    external[clear <= 0] = 0.0

    ppfd_cols = {}
    gh_factors = {gh.id: 1.0 - 0.14 * i for i, gh in enumerate(cfg.greenhouses)}
    for gh in cfg.greenhouses:
        for j, zone in enumerate(gh.zones):
            tau = np.array([transmittance(ts, gh_factors[gh.id]) for ts in index])
            inside = external * tau * (1.0 - 0.09 * j)
            if cfg.ni.applies_to(zone.treatment):
                ni_mask = np.array([cfg.ni.window.contains(ts.to_pydatetime()) for ts in index])
                inside = inside + ni_mask * cfg.lamp.ppfd_contribution
            ppfd_cols[zone.id] = np.clip(inside, 0, None)

    return SiteFrame.from_wide(pd.DataFrame(ppfd_cols, index=index),
                               pd.Series(external, index=index),
                               cfg.interval_minutes,
                               report={"synthetic": True, "purpose": "training"})


def inject_sensor_fault(frame: SiteFrame, zone_id: str, start: datetime,
                        minutes: int = 90) -> SiteFrame:
    """센서 통신 두절 구간을 심는다 (안전 조건 시연용)."""
    ppfd = frame.ppfd.copy()
    source = frame.source.copy()
    end = pd.Timestamp(start) + pd.Timedelta(minutes=minutes)
    mask = (ppfd.index >= pd.Timestamp(start)) & (ppfd.index < end)
    ppfd.loc[mask, zone_id] = np.nan
    source.loc[mask, zone_id] = "missing"
    return SiteFrame(ppfd, frame.external, source, frame.interval_minutes,
                     frame.report, frame.temperature)
