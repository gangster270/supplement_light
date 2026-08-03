"""합성 샘플 데이터 생성기 (개발·검증 전용).

실제 로거 데이터가 아직 없어도 P1~P4를 끝까지 돌려보고 검증할 수 있도록,
ZL6 export 와 **같은 형식**의 파일을 만든다 (3행 헤더, Processed 시트, 10분 간격).

★ 이 데이터는 시제품 배관 검증용이지 연구 결과가 아니다.
   실데이터가 들어오면 data/ 폴더를 교체하고 이 파일은 쓰지 않는다.

재현하는 실제 특성:
  - 태양 고도에 따른 시간대별 투과율 변화 (정오 ~50%, 아침·저녁 15~30%)
  - 온실 간 투과율 차이, 같은 온실 내 형제 구역 편차 (약 10%)
  - 날짜별 흐림 정도의 지속성 + 하루 안에서의 변동
  - NI 처리구의 야간 LED 고정 광량
  - 통신 오류로 인한 결측 구간
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTERVAL_MIN = 10

# (logger_id, 온실 투과율 계수, 형제 편차, NI 야간 PPFD)
ZONES = [
    ("z6-11111", 1.00, 1.00, 60.0),   # gh1 기준 구역 (NI)
    ("z6-11112", 1.00, 0.91, 60.0),   # gh1 형제 구역 (NI+SL) — 부분 차폐로 9% 낮음
    ("z6-22221", 0.86, 1.00, 58.0),   # gh2 기준 구역 — 피복재가 달라 투과율이 낮음
    ("z6-22222", 0.86, 0.94, 58.0),   # gh2 형제 구역
]

NI_START, NI_END = time(18, 30), time(2, 30)


def _daylength_hours(doy: int) -> float:
    """북위 약 37도의 대략적인 일장."""
    return 12.2 + 2.6 * math.sin(2 * math.pi * (doy - 80) / 365.0)


def _clear_sky_external(ts: pd.Timestamp) -> float:
    """맑은 날의 외부 PPFD (µmol·m⁻²·s⁻¹)."""
    doy = ts.dayofyear
    daylength = _daylength_hours(doy)
    solar_noon = 12.5
    sunrise, sunset = solar_noon - daylength / 2, solar_noon + daylength / 2
    hour = ts.hour + ts.minute / 60.0
    if not (sunrise < hour < sunset):
        return 0.0
    phase = (hour - sunrise) / daylength
    peak = 1500 + 450 * math.sin(2 * math.pi * (doy - 80) / 365.0)
    return max(0.0, peak * math.sin(math.pi * phase))


def _transmittance(ts: pd.Timestamp, gh_factor: float) -> float:
    """시간대별 투과율. 저고도(아침·저녁)에서 골조 차폐로 크게 낮아진다."""
    doy = ts.dayofyear
    daylength = _daylength_hours(doy)
    solar_noon = 12.5
    sunrise = solar_noon - daylength / 2
    hour = ts.hour + ts.minute / 60.0
    phase = (hour - sunrise) / daylength
    if not (0 < phase < 1):
        return 0.15 * gh_factor
    elevation = math.sin(math.pi * phase)
    return (0.14 + 0.36 * elevation) * gh_factor


def _cloud_series(index: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """날짜별 흐림 정도(지속성 있음) × 하루 안의 변동."""
    days = pd.Series(index.date, index=index)
    unique_days = sorted(days.unique())

    # 날짜별 청천지수: AR(1) — 흐린 날은 이어지는 경향이 있다
    daily = {}
    level = 0.75
    for d in unique_days:
        level = 0.72 * level + 0.28 * rng.beta(4.5, 2.0)
        daily[d] = float(np.clip(level, 0.08, 1.0))

    base = days.map(daily).to_numpy(dtype=float)
    # 하루 안의 구름 통과 (부드러운 노이즈)
    noise = rng.normal(0, 1, len(index))
    kernel = np.ones(9) / 9
    smooth = np.convolve(noise, kernel, mode="same")
    intraday = np.clip(1.0 + 0.35 * smooth, 0.25, 1.35)
    return np.clip(base * intraday, 0.02, 1.15)


def _in_ni(ts: pd.Timestamp) -> bool:
    t = ts.time()
    return t >= NI_START or t <= NI_END


def build_frames(start: str, end: str, seed: int = 20260803):
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq=f"{INTERVAL_MIN}min", inclusive="left")

    clear = np.array([_clear_sky_external(ts) for ts in index])
    cloud = _cloud_series(index, rng)
    external = np.clip(clear * cloud + rng.normal(0, 4, len(index)), 0, None)
    external[clear <= 0] = 0.0

    zone_data = {}
    for logger_id, gh_factor, sib_ratio, night_ppfd in ZONES:
        tau = np.array([_transmittance(ts, gh_factor) for ts in index])
        inside = external * tau * sib_ratio
        # NI 구간에는 LED 고정 광량이 더해진다
        ni_mask = np.array([_in_ni(ts) for ts in index])
        inside = inside + ni_mask * (night_ppfd + rng.normal(0, 1.5, len(index)))
        inside = np.clip(inside + rng.normal(0, 2.0, len(index)), 0, None)

        # 통신 오류에 의한 결측 구간을 몇 개 심는다 (실데이터에서 흔한 상황)
        values = inside.copy()
        for _ in range(rng.integers(3, 7)):
            gap_start = int(rng.integers(0, max(1, len(values) - 200)))
            gap_len = int(rng.integers(6, 180))  # 1시간 ~ 30시간
            values[gap_start:gap_start + gap_len] = np.nan
        zone_data[logger_id] = values

    return index, external, zone_data


def write_zl6_excel(path: Path, index: pd.DatetimeIndex, values: np.ndarray,
                    logger_id: str) -> None:
    """ZL6 export 형식(3행 헤더 + Processed 시트)으로 저장한다."""
    n = len(index)
    rows = [
        [logger_id, "Port 1", "Port 2"],
        [f"Records: {n}", "SQ-521", "ATMOS 14"],
        ["Timestamp", " µmol·m⁻²·s⁻¹ PPFD", " °C Air Temperature"],
    ]
    header = pd.DataFrame(rows)
    body = pd.DataFrame({
        0: index.strftime("%Y-%m-%d %H:%M:%S"),
        1: values,
        2: np.round(18 + 8 * np.sin(np.arange(n) * 2 * math.pi / 144), 2),
    })
    out = pd.concat([header, body], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="Processed Data Config 1", header=False, index=False)
        pd.DataFrame({"info": [f"synthetic sample for {logger_id}"]}).to_excel(
            writer, sheet_name="Metadata", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="합성 샘플 데이터 생성 (개발 전용)")
    ap.add_argument("--start", default="2026-03-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--out", default=str(ROOT / "data"))
    args = ap.parse_args()

    index, external, zone_data = build_frames(args.start, args.end, args.seed)
    out = Path(args.out)

    stamp = datetime.now().strftime("%d%m%y-%H%M")
    for logger_id, values in zone_data.items():
        path = out / "logger" / f"{logger_id}_{stamp}.xlsx"
        write_zl6_excel(path, index, values, logger_id)
        print(f"  {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}  ({np.isnan(values).mean():.1%} 결측)")

    ext_path = out / "external" / f"external_ppfd_{stamp}.csv"
    ext_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Timestamp": index.strftime("%Y-%m-%d %H:%M:%S"),
        "퀀텀센터(µmol·m⁻²·s⁻¹)": np.round(external, 1),
    }).to_csv(ext_path, index=False, encoding="utf-8-sig")
    print(f"  {ext_path.relative_to(ROOT) if ext_path.is_relative_to(ROOT) else ext_path}")
    print(f"\n기간 {index[0]} ~ {index[-1]}, {len(index):,}행 x {len(zone_data)}구역")


if __name__ == "__main__":
    main()
