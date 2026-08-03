"""P2: DLI 누적, 이동평균, 투과율, 결측 보정, 청천 기준선.

`agri-ppfd-gap-fill` 스킬의 원칙을 그대로 따른다:
  - 투과율은 **단일 평균이 아니라 시간대별 프로파일**로. 아침·저녁 저고도 태양광은
    골조 차폐가 커서 정오 대비 투과율이 크게 낮다 (정오 ~50%, 아침/저녁 10~30%).
  - 형제 구역 값은 그대로 대입하지 않고 **비율로 스케일링**한다.
  - 야간 고정값은 clock time 만으로 자르면 하지 계절에 자연광이 섞인다.
    반드시 **외부 PPFD 가 거의 0인 시점만** 골라 '순수 LED' 평균을 낸다.
  - 채운 값에는 항상 출처 플래그를 남긴다 (추적성).
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from .config import SiteConfig
from .ingest import SiteFrame
from .timeutil import TimeWindow, circular_doy_distance, day_of_year, minute_of_day

# 출처 플래그
SRC_MEASURED = "measured"
SRC_NIGHT = "estimated_night"
SRC_SIBLING = "estimated_sibling"
SRC_EXTERNAL = "estimated_external"
SRC_MISSING = "missing"


# =====================================================================
# DLI
# =====================================================================

def dli_from_ppfd(ppfd: pd.Series, interval_seconds: int) -> float:
    """PPFD 시계열의 적산 DLI (mol·m⁻²). 결측은 건너뛴다(0으로 보지 않는다)."""
    if ppfd is None or len(ppfd) == 0:
        return 0.0
    return float(np.nansum(ppfd.to_numpy(dtype=float)) * interval_seconds / 1e6)


def cumulative_dli(ppfd: pd.Series, interval_seconds: int) -> pd.Series:
    """시점별 누적 DLI 시계열."""
    vals = pd.Series(ppfd, dtype=float).fillna(0.0)
    return (vals.cumsum() * interval_seconds / 1e6).rename("cumulative_dli")


def daily_dli(ppfd: pd.Series, interval_seconds: int) -> pd.Series:
    """달력일 기준 일별 DLI."""
    s = pd.Series(ppfd, dtype=float)
    return (s.groupby(s.index.date).apply(lambda x: dli_from_ppfd(x, interval_seconds))
            .rename("dli"))


def moving_average(ppfd: pd.Series, window_minutes: int, interval_minutes: int,
                   min_periods: int = 1) -> pd.Series:
    """이동평균. 순간값으로 판단하면 구름 한 조각에 점·소등이 뒤집힌다."""
    window = max(1, int(round(window_minutes / max(interval_minutes, 1))))
    return pd.Series(ppfd, dtype=float).rolling(window, min_periods=min_periods).mean()


def latest_moving_average(ppfd: pd.Series, window_minutes: int, interval_minutes: int) -> float:
    """가장 최근 시점의 이동평균 값 하나. 결측뿐이면 nan."""
    if ppfd is None or len(ppfd) == 0:
        return float("nan")
    window = max(1, int(round(window_minutes / max(interval_minutes, 1))))
    tail = pd.Series(ppfd, dtype=float).tail(window)
    if tail.notna().sum() == 0:
        return float("nan")
    return float(tail.mean(skipna=True))


# =====================================================================
# 투과율 / 형제구역 비율 / 야간 기준값
# =====================================================================

def hourly_transmittance(internal: pd.Series, external: pd.Series,
                         daylight_threshold: float = 50.0,
                         exclude_periods: list[tuple] | None = None) -> dict[int, float]:
    """시간대별(0~23시) 중앙값 투과율. 반드시 그 온실 자체의 실측으로 산출한다.

    다른 온실에서 구한 비율을 재사용하면 안 된다 — 피복재·골조가 다르면 투과율이 다르다.
    """
    df = pd.DataFrame({"i": internal, "e": external}).dropna()
    mask = df["e"] > daylight_threshold
    if exclude_periods:
        for start, end in exclude_periods:
            mask &= ~((df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end)))
    sub = df[mask]
    if sub.empty:
        return {h: float("nan") for h in range(24)}
    ratio = sub["i"] / sub["e"]
    overall = float(ratio.median())
    hourly = ratio.groupby(sub.index.hour).median()
    # 데이터가 없는 시간대(새벽/심야)는 전체 중앙값으로 — 어차피 외부광이 0에 가까워 영향이 작다.
    return {h: float(hourly.get(h, overall)) for h in range(24)}


def sibling_ratio(reference: pd.Series, sibling: pd.Series,
                  reference_threshold: float = 50.0) -> float:
    """기준 구역 대비 형제 구역의 낮 시간 실측 비율 (합계비 = DLI 비와 같은 개념).

    같은 온실 안에서도 등기구·지주대 부분 차폐로 7~15% 편차가 날 수 있다.
    """
    df = pd.DataFrame({"r": reference, "s": sibling}).dropna()
    df = df[df["r"] > reference_threshold]
    if df.empty or df["r"].sum() == 0:
        return float("nan")
    return float(df["s"].sum() / df["r"].sum())


def night_baseline(zone: pd.Series, external: pd.Series, window: TimeWindow,
                   external_night_threshold: float = 2.0) -> float:
    """'순수 야간'(시각상 야간 ∩ 외부광 거의 0) 실측 평균 = 그 구역의 LED 단독 PPFD.

    이 값이 곧 `lamp.ppfd_contribution` 의 실측 근거다.
    clock time 만으로 자르면 하지에 자연광이 섞여 값이 부풀려진다.
    """
    df = pd.DataFrame({"z": zone, "e": external})
    is_clock_night = pd.Series([window.contains(ts.to_pydatetime()) for ts in df.index],
                               index=df.index)
    mask = is_clock_night & (df["e"] < external_night_threshold) & df["z"].notna()
    if mask.sum() == 0:
        return float("nan")
    return float(df.loc[mask, "z"].mean())


def remove_lamp_contribution(series: pd.Series, windows: list[TimeWindow],
                             contribution: float) -> pd.Series:
    """측정값에서 점등 기여분을 빼 '자연광만' 시계열을 복원한다 (0 미만은 0).

    ★ 청천지수·투과율·예측 평가는 **반드시** 이 값을 써야 한다.
      야간 NI 조명이 섞인 채로 청천지수를 구하면, 청천 기준선에는 없는 광량이
      분자에만 더해져 "오늘은 맑다"고 오판한다. 실제로 오전 8시 기준 청천지수가
      50%p 이상 부풀려지는 것을 확인했다 (그만큼 보광을 놓친다).
    """
    if series is None or len(series) == 0 or not windows or contribution <= 0:
        return series
    was_on = np.array([any(w.contains(ts.to_pydatetime()) for w in windows)
                       for ts in series.index])
    adjusted = series.to_numpy(dtype=float) - was_on * contribution
    return pd.Series(np.clip(adjusted, 0.0, None), index=series.index, name=series.name)


def historical_lighting_windows(cfg: SiteConfig) -> dict[str, list[TimeWindow]]:
    """구역별 '과거에 실제로 점등되어 있던 시간대' 기본 추정 = NI 스케줄.

    NI 구간 밖에 별도 보광을 운영했다면 이 추정은 불완전하다. 그 경우 실제 운영
    시간대를 직접 넘겨야 한다 (엔진·평가 함수 모두 인자로 받는다).
    """
    return {z.id: ([cfg.ni.window] if cfg.ni.applies_to(z.treatment) else [])
            for z in cfg.zones}


def natural_light_frame(frame: SiteFrame, cfg: SiteConfig,
                        windows: dict[str, list[TimeWindow]] | None = None) -> SiteFrame:
    """전 구역에서 점등 기여분을 제거한 SiteFrame (프로파일 학습·예측 평가용)."""
    windows = windows or historical_lighting_windows(cfg)
    ppfd = frame.ppfd.copy()
    for zone_id in ppfd.columns:
        ppfd[zone_id] = remove_lamp_contribution(
            ppfd[zone_id], windows.get(zone_id, []), cfg.lamp.ppfd_contribution)
    return SiteFrame(ppfd, frame.external, frame.source, frame.interval_minutes, frame.report)


def estimate_lamp_contribution(frame: SiteFrame, cfg: SiteConfig) -> dict[str, float]:
    """구역별 LED 단독 PPFD 실측 추정치.

    site.yaml 의 lamp.ppfd_contribution 을 확정할 때 쓰는 근거값.
    NI 처리구가 아닌 구역은 야간 점등이 없으므로 값이 0 근처로 나오는 게 정상이다.
    """
    out = {}
    for zone in cfg.zones:
        out[zone.id] = night_baseline(frame.ppfd[zone.id], frame.external,
                                      cfg.ni.window, cfg.external.night_threshold)
    return out


# =====================================================================
# 결측 보정
# =====================================================================

def fill_missing(frame: SiteFrame, cfg: SiteConfig,
                 exclude_periods: list[tuple] | None = None) -> SiteFrame:
    """온실 그룹별로 결측을 채운다. 우선순위를 지켜야 순환 참조 없이 전체가 채워진다.

    1. 원본값이 있으면 유지
    2. 야간 결측 → 그 구역의 순수 야간 평균
    3. 주간 결측, 기준구역 값 있음 → 기준구역값 × 형제구역 비율
    4. 기준구역 자체가 결측 → 외부 PPFD × 해당 시간대 투과율
    """
    ppfd = frame.ppfd.copy()
    source = frame.source.copy()
    external = frame.external
    idx = ppfd.index

    is_night = pd.Series([cfg.ni.window.contains(ts.to_pydatetime()) for ts in idx], index=idx)
    hours = pd.Series(idx.hour, index=idx)
    diagnostics: dict[str, dict] = {}

    for gh in cfg.greenhouses:
        ref = gh.reference_zone
        ratios = hourly_transmittance(ppfd[ref.id], external,
                                      cfg.external.daylight_threshold, exclude_periods)
        ext_est = external * hours.map(ratios)
        night_ref = night_baseline(ppfd[ref.id], external, cfg.ni.window,
                                   cfg.external.night_threshold)

        # --- 기준 구역 먼저 ---
        missing = ppfd[ref.id].isna()
        night_gap = missing & is_night & np.isfinite(night_ref)
        ppfd.loc[night_gap, ref.id] = night_ref
        source.loc[night_gap, ref.id] = SRC_NIGHT

        missing = ppfd[ref.id].isna()
        ext_gap = missing & ext_est.notna()
        ppfd.loc[ext_gap, ref.id] = ext_est[ext_gap]
        source.loc[ext_gap, ref.id] = SRC_EXTERNAL

        diagnostics[gh.id] = {
            "reference_zone": ref.id,
            "hourly_transmittance": ratios,
            "night_baseline": {ref.id: night_ref},
            "sibling_ratio": {},
        }

        # --- 형제 구역 (이미 채워진 기준구역 값을 사용) ---
        for sib in gh.sibling_zones:
            ratio = sibling_ratio(frame.ppfd[ref.id], frame.ppfd[sib.id],
                                  cfg.external.daylight_threshold)
            night_sib = night_baseline(ppfd[sib.id], external, cfg.ni.window,
                                       cfg.external.night_threshold)
            diagnostics[gh.id]["sibling_ratio"][sib.id] = ratio
            diagnostics[gh.id]["night_baseline"][sib.id] = night_sib

            missing = ppfd[sib.id].isna()
            night_gap = missing & is_night & np.isfinite(night_sib)
            ppfd.loc[night_gap, sib.id] = night_sib
            source.loc[night_gap, sib.id] = SRC_NIGHT

            if np.isfinite(ratio):
                missing = ppfd[sib.id].isna()
                sib_gap = missing & ppfd[ref.id].notna()
                ppfd.loc[sib_gap, sib.id] = ppfd.loc[sib_gap, ref.id] * ratio
                source.loc[sib_gap, sib.id] = SRC_SIBLING

    report = dict(frame.report)
    report["fill"] = diagnostics
    return SiteFrame(ppfd, external, source, frame.interval_minutes, report)


# =====================================================================
# 청천(clear-sky) 기준선
# =====================================================================

class ClearSkyProfile:
    """'맑은 날이었다면 이 시각에 얼마였을까'의 기준선.

    천문 계산 대신 **그 사이트의 과거 실측 상위 분위수**로 만든다. 이러면 위경도·
    피복재·차광 구조가 자동으로 반영되고, 외부 라이브러리 의존도 생기지 않는다.
    대신 사이트 전용이라 다른 온실로 옮기면 다시 학습시켜야 한다.

    lamp_masked_by: 이 시계열이 임계값 미만인 시점(=야간)은 학습에서 제외한다.
        내부 PPFD로 프로파일을 만들 때 야간 보광이 섞여 들어가는 것을 막는다.
    """

    def __init__(self, series: pd.Series, interval_minutes: int, quantile: float = 0.9,
                 doy_window_days: int = 15, lamp_masked_by: pd.Series | None = None,
                 mask_threshold: float = 2.0):
        s = pd.Series(series, dtype=float).dropna()
        if lamp_masked_by is not None:
            ext = pd.Series(lamp_masked_by, dtype=float).reindex(s.index)
            s = s[(ext.isna()) | (ext >= mask_threshold)]
        if s.empty:
            raise ValueError("청천 프로파일을 만들 데이터가 없습니다.")
        self.interval_minutes = interval_minutes
        self.quantile = float(quantile)
        self.doy_window_days = int(doy_window_days)
        self._values = s.to_numpy(dtype=float)
        # 벡터 연산을 위해 numpy 배열로 보관한다. 파이썬 루프로 돌리면 1년치
        # 데이터에서 날짜마다 수만 번 반복되어 실행이 수 분 단위로 느려진다.
        self._doy = np.array([day_of_year(ts) for ts in s.index], dtype=np.int32)
        self._mod = np.array([minute_of_day(ts.to_pydatetime()) for ts in s.index],
                             dtype=np.int32)
        self._cache: dict[int, pd.Series] = {}
        self._cum_cache: dict[int, np.ndarray] = {}
        self._grid = np.arange(0, 24 * 60, max(1, interval_minutes))

    def for_date(self, target: date | datetime) -> pd.Series:
        """해당 날짜의 청천 기준 PPFD (index=minute_of_day)."""
        doy = day_of_year(target)
        if doy in self._cache:
            return self._cache[doy]
        diff = np.abs(self._doy - doy)
        dist = np.minimum(diff, 365 - diff)          # 연도 경계를 넘는 거리
        sel = dist <= self.doy_window_days
        if not sel.any():                            # 계절 데이터가 부족하면 전체 기간으로
            sel = np.ones_like(dist, dtype=bool)
        sub = pd.DataFrame({"mod": self._mod[sel], "value": self._values[sel]})
        profile = sub.groupby("mod")["value"].quantile(self.quantile)
        profile = profile.reindex(self._grid).interpolate().fillna(0.0).clip(lower=0.0)
        self._cache[doy] = profile
        return profile

    def _cumulative(self, doy_key: int, target: date | datetime) -> np.ndarray:
        """격자 시작부터의 누적 DLI (mol). dli_between 을 O(1)로 만든다."""
        if doy_key not in self._cum_cache:
            profile = self.for_date(target).to_numpy(dtype=float)
            step = self.interval_minutes * 60 / 1e6
            self._cum_cache[doy_key] = np.concatenate([[0.0], np.cumsum(profile) * step])
        return self._cum_cache[doy_key]

    def dli_between(self, target: date | datetime, start_minute: int, end_minute: int) -> float:
        """[start_minute, end_minute) 구간의 청천 기준 적산 DLI (mol·m⁻²)."""
        if end_minute <= start_minute:
            return 0.0
        cum = self._cumulative(day_of_year(target), target)
        n = len(cum) - 1
        i = int(np.clip(np.ceil(start_minute / self.interval_minutes), 0, n))
        j = int(np.clip(np.ceil(end_minute / self.interval_minutes), 0, n))
        return float(cum[j] - cum[i])

    def remaining_dli(self, moment: datetime) -> float:
        """moment 부터 그날 자정까지 남은 청천 기준 DLI."""
        return self.dli_between(moment, minute_of_day(moment), 24 * 60)

    def elapsed_dli(self, moment: datetime) -> float:
        """그날 00:00 부터 moment 까지의 청천 기준 DLI."""
        return self.dli_between(moment, 0, minute_of_day(moment))


def clearness_index(actual: pd.Series, profile: ClearSkyProfile, moment: datetime,
                    interval_seconds: int, min_samples: int = 6,
                    min_reference_dli: float = 0.5) -> float | None:
    """오늘의 흐림 정도 (0~1+). 지금까지의 실측 적산 / 같은 구간의 청천 기준 적산.

    ★ actual 은 반드시 `remove_lamp_contribution()` 을 거친 '자연광만' 시계열이어야 한다.

    None 을 반환하는 경우 = 아직 판단할 근거가 없음 (해뜨기 전, 기준 적산이 너무 작음).
    호출측은 이때 임의로 1.0 을 가정하지 말고 '불확실'로 표시해야 한다.

    min_reference_dli: 이른 아침에는 기준 적산이 작아 분모가 조금만 흔들려도 비율이
        폭발한다. 기준 적산이 이 값에 못 미치면 청천지수를 계산하지 않는다.
    """
    reference = profile.elapsed_dli(moment)
    if reference < max(min_reference_dli, 0.05):
        return None
    valid = pd.Series(actual, dtype=float).notna().sum()
    if valid < min_samples:
        return None
    measured = dli_from_ppfd(actual, interval_seconds)
    return float(max(measured / reference, 0.0))


def build_zone_profiles(frame: SiteFrame, cfg: SiteConfig) -> dict[str, ClearSkyProfile]:
    """구역별 청천 프로파일. 야간 보광이 섞이지 않도록 외부광 기준으로 마스킹한다."""
    out = {}
    has_external = frame.external.notna().any()
    for zone in cfg.zones:
        out[zone.id] = ClearSkyProfile(
            frame.ppfd[zone.id], frame.interval_minutes,
            quantile=cfg.forecast.clear_sky_quantile,
            doy_window_days=cfg.forecast.doy_window_days,
            lamp_masked_by=frame.external if has_external else None,
            mask_threshold=cfg.external.night_threshold,
        )
    return out
