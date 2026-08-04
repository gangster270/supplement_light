"""P3: 잔여 자연광 DLI 예측.

기본 가정: **오늘 남은 시간의 흐림 정도가 지금까지와 비슷하다.**
    예상 잔여 = (청천 기준 잔여 DLI) × (오늘 지금까지의 청천지수 CI) × (모드 계수)

이 가정은 오전에 맑다가 오후에 흐려지는 날 깨진다. 그래서 점 추정만 내지 않고,
**그 가정이 과거에 얼마나 틀렸는지를 데이터로 학습해서 밴드(low~high)로 함께 낸다.**
UI는 이 밴드를 반드시 표시해야 한다 — 예측을 확정값처럼 보여주면 안 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from .config import SiteConfig
from .ingest import SiteFrame
from .metrics import ClearSkyProfile, clearness_index, dli_from_ppfd
from .timeutil import minute_of_day

CONFIDENCE_HIGH = "high"
CONFIDENCE_LOW = "low"


@dataclass(frozen=True)
class Forecast:
    """잔여 자연광 예측 결과."""

    expected: float               # 예상 잔여 자연광 DLI (mol·m⁻²)
    low: float                    # 하한 (흐려질 경우)
    high: float                   # 상한 (맑아질 경우)
    clearness: float | None       # 오늘 지금까지의 청천지수 (None = 판단 불가)
    clear_sky_remaining: float    # 맑았을 때의 잔여 DLI
    confidence: str               # high | low
    note: str = ""

    @property
    def band_width(self) -> float:
        return max(self.high - self.low, 0.0)


N_PROGRESS_BINS = 10


def progress_bin(progress: float) -> int:
    """청천 진행률(0~1)을 구간 번호로."""
    return int(min(max(progress, 0.0), 0.999) * N_PROGRESS_BINS)


@dataclass
class ClearnessDynamics:
    """'지금까지의 흐림 정도'가 '남은 시간의 흐림 정도'를 얼마나 잘 맞히는지의 과거 분포.

    r = (남은 시간 실제 CI) / (지금까지의 CI). r=1 이면 가정이 맞은 날, r<1 이면 오후에 더 흐려진 날.

    ★ 시각(hour)이 아니라 **청천 진행률**(그날 청천 기준 적산 중 지금까지의 비율)로
      조건화한다. 같은 08시라도 3월은 일출 1시간 후, 5월은 3시간 후라 태양 진행도가
      전혀 다르다. 시각으로 학습하면 계절이 바뀔 때 보정 방향이 뒤집힌다
      (실측: 시각 기준으로는 보정이 오히려 편의를 +0.6 → +1.3 mol 로 키웠다).

    q10~q90 이 예측 밴드가 된다. 중앙값(calibration)은 진단용으로만 보관한다 —
    이걸 보정계수로 곱해 봤으나 오차가 오히려 커져서 적용하지 않는다 (Forecaster.predict 참고).
    """

    by_bin: dict[int, tuple[float, float]] = field(default_factory=dict)
    overall: tuple[float, float] = (0.5, 1.5)
    median_by_bin: dict[int, float] = field(default_factory=dict)
    median_overall: float = 1.0
    n_samples: int = 0

    def band(self, progress: float) -> tuple[float, float]:
        return self.by_bin.get(progress_bin(progress), self.overall)

    def calibration(self, progress: float) -> float:
        """계통 편의 보정계수. 학습 데이터가 없으면 1.0 (보정 안 함)."""
        if self.n_samples == 0:
            return 1.0
        return self.median_by_bin.get(progress_bin(progress), self.median_overall)


def learn_clearness_dynamics(frame: SiteFrame, profile: ClearSkyProfile, zone_id: str,
                             cfg: SiteConfig, min_samples_per_bin: int = 8,
                             quantiles: tuple[float, float] = (0.1, 0.9),
                             sample_minutes: int = 30) -> ClearnessDynamics:
    """과거 데이터에서 CI 지속성 가정의 오차 분포를 청천 진행률별로 학습한다.

    학습에 쓰는 것은 **과거 구간 전체**다 (판단 루프 밖의 '사전 준비' 단계).
    실시간 판단 중에는 절대 미래를 보지 않는다.
    """
    series = frame.ppfd[zone_id]
    interval_s = frame.interval_seconds
    step = max(sample_minutes, frame.interval_minutes)
    records: list[tuple[float, float]] = []

    for day, day_series in series.groupby(series.index.date):
        if day_series.notna().sum() < 6:
            continue
        clear_day = profile.dli_between(day, 0, 24 * 60)
        if clear_day <= 0.5:
            continue
        # 하루치 누적 DLI를 한 번만 계산해 두고 인덱싱으로 자른다
        # (샘플마다 시계열을 슬라이싱하면 1년치에서 수백만 번 반복된다).
        minutes = np.array([minute_of_day(ts.to_pydatetime()) for ts in day_series.index])
        cum = np.concatenate([[0.0],
                              np.nancumsum(day_series.to_numpy(dtype=float)) * interval_s / 1e6])
        total_actual = cum[-1]
        for minute in range(0, 24 * 60, step):
            clear_sofar = profile.dli_between(day, 0, minute)
            clear_rest = clear_day - clear_sofar
            if clear_sofar <= 0.3 or clear_rest <= 0.3:
                continue
            k = int(np.searchsorted(minutes, minute, side="left"))
            ci_sofar = cum[k] / clear_sofar
            ci_rest = (total_actual - cum[k]) / clear_rest
            if ci_sofar <= 0.05:  # 기준이 너무 작으면 비율이 폭발한다
                continue
            records.append((clear_sofar / clear_day, ci_rest / ci_sofar))

    if not records:
        return ClearnessDynamics()

    df = pd.DataFrame(records, columns=["progress", "ratio"])
    df["bin"] = df["progress"].map(progress_bin)
    lo_q, hi_q = quantiles
    overall = (float(df["ratio"].quantile(lo_q)), float(df["ratio"].quantile(hi_q)))
    by_bin, median_by_bin = {}, {}
    for b, grp in df.groupby("bin"):
        if len(grp) >= min_samples_per_bin:
            by_bin[int(b)] = (float(grp["ratio"].quantile(lo_q)),
                              float(grp["ratio"].quantile(hi_q)))
            median_by_bin[int(b)] = float(grp["ratio"].median())
    return ClearnessDynamics(by_bin=by_bin, overall=overall,
                             median_by_bin=median_by_bin,
                             median_overall=float(df["ratio"].median()),
                             n_samples=len(df))


class Forecaster:
    """구역 하나에 대한 잔여 자연광 예측기."""

    def __init__(self, profile: ClearSkyProfile, cfg: SiteConfig,
                 dynamics: ClearnessDynamics | None = None,
                 default_clearness: float = 0.7):
        self.profile = profile
        self.cfg = cfg
        self.dynamics = dynamics or ClearnessDynamics()
        # CI를 계산할 수 없을 때(해뜨기 전 등) 쓰는 기본값.
        # 1.0(맑음)으로 두면 "자연광으로 충분하다"고 낙관해 보광을 놓친다. 보수적으로 잡는다.
        self.default_clearness = float(default_clearness)

    def predict(self, today_series: pd.Series, moment: datetime,
                mode: str | None = None) -> Forecast:
        """moment 시점에서 그날 남은 시간의 자연광 DLI를 예측한다.

        today_series 는 **당일 00:00 ~ moment 까지만** 넘겨야 한다 (미래 금지).
        """
        fc_cfg = self.cfg.forecast if mode is None else self.cfg.forecast.with_mode(mode)
        clear_remaining = self.profile.remaining_dli(moment)

        ci = clearness_index(today_series, self.profile, moment,
                             self.cfg.interval_seconds, fc_cfg.min_clearness_samples,
                             fc_cfg.min_reference_dli)

        if clear_remaining <= 0.01:
            # 이미 해가 진 뒤 — 남은 자연광은 없다. 이건 불확실한 게 아니라 확실한 0이다.
            return Forecast(0.0, 0.0, 0.0, ci, clear_remaining, CONFIDENCE_HIGH,
                            "일몰 이후 — 남은 자연광 없음")

        if ci is None:
            expected = clear_remaining * self.default_clearness * fc_cfg.factor
            return Forecast(
                expected=expected,
                low=clear_remaining * 0.1,
                high=clear_remaining * 1.0,
                clearness=None,
                clear_sky_remaining=clear_remaining,
                confidence=CONFIDENCE_LOW,
                note=(f"오늘의 흐림 정도를 아직 판단할 수 없어 기본값 "
                      f"{self.default_clearness:.0%}를 가정했습니다 (불확실)."))

        elapsed = self.profile.elapsed_dli(moment)
        clear_day = elapsed + clear_remaining
        progress = elapsed / clear_day if clear_day > 0 else 0.0
        base = clear_remaining * ci
        expected = base * fc_cfg.factor
        lo_r, hi_r = self.dynamics.band(progress)
        low, high = base * lo_r, base * hi_r
        confidence = CONFIDENCE_HIGH if self.dynamics.n_samples > 0 else CONFIDENCE_LOW
        note = (f"오늘 지금까지 청천 대비 {ci:.0%}. 남은 시간도 비슷하다고 가정했습니다."
                if self.dynamics.n_samples
                else f"오늘 지금까지 청천 대비 {ci:.0%} (과거 오차 분포 미학습 — 밴드는 참고용).")
        return Forecast(expected, low, high, ci, clear_remaining, confidence, note)

    # 학습된 중앙값(dynamics.calibration)을 곱해 계통편의를 보정하는 방안을 시험했으나
    # **오차가 오히려 커져서 적용하지 않는다** (실측: bias +0.60 → +1.11 mol).
    # 이유: 중앙값은 '프로파일이 학습한 계절'에서의 치우침이고, 실제 예측은
    # 아직 학습하지 못한 계절로 외삽하는 상황이라 치우침의 방향이 다르다.
    # dynamics.calibration() 은 진단용으로 남겨 둔다 (밴드 산출에는 계속 쓰인다).


def build_forecasters(frame: SiteFrame, profiles: dict[str, ClearSkyProfile],
                      cfg: SiteConfig) -> dict[str, Forecaster]:
    """구역별 예측기를 한 번에 준비한다 (앱/백테스트 시작 시 1회)."""
    return {
        zone_id: Forecaster(profile, cfg,
                            learn_clearness_dynamics(frame, profile, zone_id, cfg))
        for zone_id, profile in profiles.items()
    }


def evaluate_forecast(frame: SiteFrame, forecaster: Forecaster, zone_id: str,
                      cfg: SiteConfig, hours: tuple[int, ...] = (8, 10, 12, 14, 16)
                      ) -> pd.DataFrame:
    """고정된 예측기 하나로 정확도를 측정한다 (P3 검증용).

    ★ frame 은 반드시 `natural_light_frame()` 을 거친 자연광 프레임이어야 한다.
    ★ 예측기를 학습한 기간과 평가 기간의 계절이 다르면 결과가 왜곡된다.
      청천 프로파일은 본 적 없는 계절로 외삽하지 못한다 (봄 프로파일로 여름을 못 맞힌다).
      운영 상황을 반영한 측정은 `walk_forward_evaluate()` 를 쓸 것.

    반환: date, hour, predicted, actual, error, in_band
    """
    series = frame.ppfd[zone_id]
    rows = []
    for day, day_series in series.groupby(series.index.date):
        if day_series.notna().sum() < 6:
            continue
        for hour in hours:
            cut = pd.Timestamp(day) + pd.Timedelta(hours=hour)
            sofar = day_series[day_series.index < cut]
            rest = day_series[day_series.index >= cut]
            if len(rest) == 0:
                continue
            fc = forecaster.predict(sofar, cut.to_pydatetime())
            actual = dli_from_ppfd(rest, frame.interval_seconds)
            rows.append({
                "date": day, "hour": hour,
                "predicted": fc.expected, "actual": actual,
                "error": fc.expected - actual,
                "low": fc.low, "high": fc.high,
                "in_band": bool(fc.low <= actual <= fc.high),
                "clearness": fc.clearness,
            })
    return _attach_metrics(pd.DataFrame(rows))


def _attach_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty:
        df.attrs["mae"] = float(df["error"].abs().mean())
        df.attrs["bias"] = float(df["error"].mean())
        df.attrs["band_coverage"] = float(df["in_band"].mean())
        denom = df["predicted"].replace(0.0, np.nan)
        df.attrs["mape"] = float((df["error"].abs() / denom).mean())
    return df


def build_single_forecaster(frame: SiteFrame, zone_id: str, cfg: SiteConfig) -> Forecaster:
    """구역 하나의 청천 프로파일 + 예측기를 만든다 (walk-forward 재학습용).

    frame 은 자연광 프레임이어야 한다.
    """
    profile = ClearSkyProfile(
        frame.ppfd[zone_id], frame.interval_minutes,
        quantile=cfg.forecast.clear_sky_quantile,
        doy_window_days=cfg.forecast.doy_window_days,
        lamp_masked_by=frame.external if frame.external.notna().any() else None,
        mask_threshold=cfg.external.night_threshold,
    )
    return Forecaster(profile, cfg, learn_clearness_dynamics(frame, profile, zone_id, cfg))


def walk_forward_evaluate(natural_frame: SiteFrame, zone_id: str, cfg: SiteConfig,
                          start: datetime | pd.Timestamp,
                          retrain_days: int = 7, min_train_days: int = 30,
                          hours: tuple[int, ...] = (8, 10, 12, 14, 16)) -> pd.DataFrame:
    """운영 상황을 그대로 재현한 예측 정확도 측정 (P3의 정식 검증 방법).

    실제 운영에서는 청천 프로파일을 주기적으로 다시 학습한다. 그러므로 평가도
    **매 시점 그 이전 데이터만으로 학습한 예측기**로 해야 한다.
    한 번 학습한 예측기를 계절이 다른 구간에 고정 적용하면 실제보다 나쁘게 나온다.

    retrain_days: 며칠마다 재학습할지 (운영 시 배치 주기와 맞추면 된다)
    min_train_days: 최소 학습 일수. 이보다 짧으면 그 날짜는 평가하지 않는다.
    """
    series = natural_frame.ppfd[zone_id]
    by_day = {d: g for d, g in series.groupby(series.index.date)}
    all_days = sorted(by_day)
    start_date = pd.Timestamp(start).date()
    eval_days = [d for d in all_days if d >= start_date]

    rows: list[dict] = []
    forecaster: Forecaster | None = None
    last_trained: date | None = None

    for day in eval_days:
        n_train_days = sum(1 for d in all_days if d < day)
        if n_train_days < min_train_days:
            continue
        if forecaster is None or last_trained is None or (day - last_trained).days >= retrain_days:
            training = natural_frame.slice(end=pd.Timestamp(day))
            forecaster = build_single_forecaster(training, zone_id, cfg)
            last_trained = day

        day_series = by_day[day]
        if day_series.notna().sum() < 6:
            continue
        for hour in hours:
            cut = pd.Timestamp(day) + pd.Timedelta(hours=hour)
            sofar = day_series[day_series.index < cut]
            rest = day_series[day_series.index >= cut]
            if len(rest) == 0:
                continue
            fc = forecaster.predict(sofar, cut.to_pydatetime())
            actual = dli_from_ppfd(rest, natural_frame.interval_seconds)
            rows.append({
                "date": day, "hour": hour, "trained_at": last_trained,
                "predicted": fc.expected, "actual": actual,
                "error": fc.expected - actual, "low": fc.low, "high": fc.high,
                "in_band": bool(fc.low <= actual <= fc.high),
                "clearness": fc.clearness, "confidence": fc.confidence,
            })
    return _attach_metrics(pd.DataFrame(rows))
