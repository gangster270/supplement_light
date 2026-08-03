"""site.yaml 로드 / 검증 / 미확정 파라미터 추적.

설계 원칙:
  - 사이트 상수는 코드에 하드코딩하지 않는다. 전부 이 모듈을 통해서만 들어온다.
  - 확인받지 않은 값(verified: false)은 실행을 막지는 않되 반드시 드러낸다.
    조용히 기본값으로 굴러가면 연구 결과가 왜곡된 줄도 모르고 쓰게 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .timeutil import TimeWindow, parse_time

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "site.yaml"


# =====================================================================
# 개별 설정 블록
# =====================================================================

@dataclass(frozen=True)
class ZoneConfig:
    id: str
    name: str
    treatment: str
    greenhouse_id: str
    greenhouse_name: str
    reference: bool
    logger_id: str
    ppfd_column: str


@dataclass(frozen=True)
class GreenhouseConfig:
    id: str
    name: str
    zones: list[ZoneConfig]

    @property
    def reference_zone(self) -> ZoneConfig:
        for z in self.zones:
            if z.reference:
                return z
        # 기준 구역이 지정되지 않았으면 첫 구역을 쓰되, 검증 단계에서 경고를 남긴다.
        return self.zones[0]

    @property
    def sibling_zones(self) -> list[ZoneConfig]:
        ref = self.reference_zone
        return [z for z in self.zones if z.id != ref.id]


@dataclass(frozen=True)
class LampConfig:
    power_w_per_fixture: float
    fixtures_per_zone: int
    ppfd_contribution: float

    @property
    def power_kw_per_zone(self) -> float:
        return self.power_w_per_fixture * self.fixtures_per_zone / 1000.0

    def dli_per_hour(self) -> float:
        """등기구를 1시간 켰을 때 구역 평균이 얻는 DLI(mol·m⁻²)."""
        return self.ppfd_contribution * 3600.0 / 1e6


@dataclass(frozen=True)
class NIConfig:
    enabled: bool
    window: TimeWindow
    treatments: list[str]

    def applies_to(self, treatment: str) -> bool:
        return self.enabled and treatment in self.treatments


@dataclass(frozen=True)
class DecisionConfig:
    target_dli: float
    on_threshold: float
    off_threshold: float
    moving_average_minutes: int
    min_on_minutes: int
    min_off_minutes: int
    max_daily_lighting_hours: float
    max_photoperiod_hours: float
    allowed_windows: list[TimeWindow]
    urgency_margin_hours: float


@dataclass(frozen=True)
class ForecastConfig:
    mode: str
    mode_factors: dict[str, float]
    clear_sky_quantile: float
    doy_window_days: int
    min_clearness_samples: int
    min_reference_dli: float = 0.5

    @property
    def factor(self) -> float:
        return float(self.mode_factors.get(self.mode, 1.0))

    def with_mode(self, mode: str) -> "ForecastConfig":
        return ForecastConfig(mode, self.mode_factors, self.clear_sky_quantile,
                              self.doy_window_days, self.min_clearness_samples,
                              self.min_reference_dli)


@dataclass(frozen=True)
class TariffSlot:
    name: str
    window: TimeWindow
    won_per_kwh: float


@dataclass(frozen=True)
class TariffSeason:
    name: str
    months: list[int]
    slots: list[TariffSlot]


@dataclass(frozen=True)
class TariffConfig:
    currency: str
    seasons: list[TariffSeason]


@dataclass(frozen=True)
class ExternalConfig:
    ppfd_column: str
    night_threshold: float
    daylight_threshold: float


@dataclass(frozen=True)
class SiteConfig:
    name: str
    crop: str
    timezone: str
    interval_minutes: int
    greenhouses: list[GreenhouseConfig]
    external: ExternalConfig
    lamp: LampConfig
    ni: NIConfig
    decision: DecisionConfig
    forecast: ForecastConfig
    tariff: TariffConfig
    logger_dir: Path
    external_dir: Path
    unverified_groups: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # --- 조회 헬퍼 -----------------------------------------------------
    @property
    def interval_seconds(self) -> int:
        return self.interval_minutes * 60

    @property
    def zones(self) -> list[ZoneConfig]:
        return [z for gh in self.greenhouses for z in gh.zones]

    @property
    def zone_ids(self) -> list[str]:
        return [z.id for z in self.zones]

    def zone(self, zone_id: str) -> ZoneConfig:
        for z in self.zones:
            if z.id == zone_id:
                return z
        raise KeyError(f"알 수 없는 구역 id: {zone_id!r} (사용 가능: {self.zone_ids})")

    def greenhouse_of(self, zone_id: str) -> GreenhouseConfig:
        for gh in self.greenhouses:
            if any(z.id == zone_id for z in gh.zones):
                return gh
        raise KeyError(f"구역 {zone_id!r} 이 속한 온실을 찾을 수 없습니다.")

    @property
    def is_fully_verified(self) -> bool:
        return not self.unverified_groups

    def unverified_message(self) -> str:
        if self.is_fully_verified:
            return ""
        return ("미확정 파라미터: " + ", ".join(self.unverified_groups) +
                " — 임시값으로 동작 중입니다. 실제 판단에 쓰기 전에 config/site.yaml 에서 "
                "실측값으로 교체하고 verified: true 로 바꾸세요.")


# =====================================================================
# 로더
# =====================================================================

def _parse_windows(items: list[str]) -> list[TimeWindow]:
    return [TimeWindow.parse(t, label=t) for t in items]


def _parse_greenhouses(block: dict, warnings: list[str]) -> list[GreenhouseConfig]:
    out = []
    for gh in block.get("items", []):
        zones = []
        for z in gh.get("zones", []):
            src = z.get("source", {}) or {}
            zones.append(ZoneConfig(
                id=z["id"],
                name=z.get("name", z["id"]),
                treatment=z.get("treatment", ""),
                greenhouse_id=gh["id"],
                greenhouse_name=gh.get("name", gh["id"]),
                reference=bool(z.get("reference", False)),
                logger_id=src.get("logger_id", ""),
                ppfd_column=src.get("ppfd_column", "PPFD"),
            ))
        if not zones:
            warnings.append(f"온실 {gh['id']!r} 에 구역이 하나도 없습니다.")
            continue
        n_ref = sum(1 for z in zones if z.reference)
        if n_ref == 0:
            warnings.append(
                f"온실 {gh['id']!r} 에 기준 구역(reference: true)이 없어 첫 구역을 기준으로 씁니다.")
        elif n_ref > 1:
            warnings.append(f"온실 {gh['id']!r} 에 기준 구역이 {n_ref}개입니다. 1개만 지정하세요.")
        out.append(GreenhouseConfig(id=gh["id"], name=gh.get("name", gh["id"]), zones=zones))
    if not out:
        raise ValueError("greenhouses.items 가 비어 있습니다. 온실-구역 매핑을 설정하세요.")
    return out


def _parse_tariff(block: dict) -> TariffConfig:
    seasons = []
    for s in block.get("seasons", []):
        slots = [TariffSlot(name=sl.get("name", ""),
                            window=TimeWindow.parse(sl["hours"], label=sl.get("name", "")),
                            won_per_kwh=float(sl["won_per_kwh"]))
                 for sl in s.get("slots", [])]
        seasons.append(TariffSeason(name=s.get("name", ""),
                                    months=[int(m) for m in s.get("months", [])],
                                    slots=slots))
    return TariffConfig(currency=block.get("currency", "KRW"), seasons=seasons)


def _validate(cfg: SiteConfig) -> None:
    """물리적으로 말이 안 되는 조합을 초기에 잡는다."""
    d = cfg.decision
    if d.on_threshold >= d.off_threshold:
        raise ValueError(
            f"히스테리시스가 성립하지 않습니다: on_threshold({d.on_threshold}) < "
            f"off_threshold({d.off_threshold}) 여야 합니다. "
            "같거나 뒤집히면 임계값 근처에서 점·소등이 반복됩니다.")
    if d.target_dli <= 0:
        raise ValueError("decision.target_dli 는 0보다 커야 합니다.")
    if d.max_daily_lighting_hours <= 0:
        raise ValueError("decision.max_daily_lighting_hours 는 0보다 커야 합니다.")
    if cfg.lamp.ppfd_contribution <= 0:
        raise ValueError(
            "lamp.ppfd_contribution 이 0 이하입니다. 등기구 점등 시 구역 평균 PPFD 기여량이 "
            "없으면 필요 점등시간을 계산할 수 없습니다.")
    if cfg.interval_minutes <= 0:
        raise ValueError("site.interval_minutes 는 0보다 커야 합니다.")
    if not cfg.decision.allowed_windows:
        cfg.warnings.append(
            "decision.allowed_windows 가 비어 있습니다. NI 구간 외에는 점등 권고가 나오지 않습니다.")

    # NI 는 강제 점등이므로 일일 점등 상한을 먼저 갉아먹는다. NI 만으로 상한이 차면
    # 보광 판단이 항상 상한에 막혀 서비스가 사실상 동작하지 않는다.
    if cfg.ni.enabled:
        ni_minutes = _minutes_in_day(cfg.ni.window, cfg.interval_minutes)
        cap_minutes = d.max_daily_lighting_hours * 60
        if cap_minutes <= ni_minutes:
            cfg.warnings.append(
                f"NI 스케줄({cfg.ni.window})만으로 하루 약 {ni_minutes / 60:.1f}시간이 점등되는데 "
                f"일일 점등 상한이 {d.max_daily_lighting_hours:.1f}시간입니다. "
                f"보광 판단이 항상 상한에 막힙니다 — max_daily_lighting_hours 를 "
                f"{ni_minutes / 60:.1f}시간보다 크게 잡으세요.")
        elif cap_minutes - ni_minutes < 60:
            cfg.warnings.append(
                f"NI 를 제외하면 재량 점등 여유가 "
                f"{(cap_minutes - ni_minutes) / 60:.1f}시간뿐입니다.")


def _minutes_in_day(window: TimeWindow, interval_minutes: int) -> float:
    """달력일 하루 안에서 시간창이 차지하는 분. 자정을 넘는 창도 하루분만 센다."""
    from datetime import datetime, timedelta
    base = datetime(2026, 1, 1)
    step = timedelta(minutes=interval_minutes)
    count, cur = 0, base
    while cur < base + timedelta(days=1):
        if window.contains(cur):
            count += 1
        cur += step
    return count * interval_minutes


def load_config(path: str | Path | None = None) -> SiteConfig:
    """site.yaml 을 읽어 SiteConfig 로 만든다."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    warnings: list[str] = []
    unverified: list[str] = []

    def check_verified(block: dict, label: str) -> dict:
        block = block or {}
        if not block.get("verified", False):
            unverified.append(label)
        return block

    site = raw.get("site", {}) or {}
    gh_block = check_verified(raw.get("greenhouses", {}), "온실·구역 매핑")
    lamp_block = check_verified(raw.get("lamp", {}), "등기구 사양")
    ni_block = check_verified(raw.get("ni", {}), "NI 스케줄")
    dec_block = check_verified(raw.get("decision", {}), "판단 기준(목표 DLI·임계값)")
    tariff_block = check_verified(raw.get("tariff", {}), "전기요금 단가")
    ext_block = raw.get("external", {}) or {}
    fc_block = raw.get("forecast", {}) or {}
    data_block = raw.get("data", {}) or {}

    root = path.resolve().parent.parent

    cfg = SiteConfig(
        name=site.get("name", "실험 사이트"),
        crop=site.get("crop", ""),
        timezone=site.get("timezone", "Asia/Seoul"),
        interval_minutes=int(site.get("interval_minutes", 10)),
        greenhouses=_parse_greenhouses(gh_block, warnings),
        external=ExternalConfig(
            ppfd_column=ext_block.get("ppfd_column", "PPFD"),
            night_threshold=float(ext_block.get("night_threshold", 2.0)),
            daylight_threshold=float(ext_block.get("daylight_threshold", 50.0)),
        ),
        lamp=LampConfig(
            power_w_per_fixture=float(lamp_block.get("power_w_per_fixture", 0.0)),
            fixtures_per_zone=int(lamp_block.get("fixtures_per_zone", 0)),
            ppfd_contribution=float(lamp_block.get("ppfd_contribution", 0.0)),
        ),
        ni=NIConfig(
            enabled=bool(ni_block.get("enabled", False)),
            window=TimeWindow(
                start=parse_time(ni_block.get("start", "18:30")),
                end=parse_time(ni_block.get("end", "02:30")),
                include_start=bool(ni_block.get("include_start", True)),
                include_end=bool(ni_block.get("include_end", True)),
                label="NI",
            ),
            treatments=list(ni_block.get("treatments", [])),
        ),
        decision=DecisionConfig(
            target_dli=float(dec_block.get("target_dli", 12.0)),
            on_threshold=float(dec_block.get("on_threshold", 150.0)),
            off_threshold=float(dec_block.get("off_threshold", 250.0)),
            moving_average_minutes=int(dec_block.get("moving_average_minutes", 30)),
            min_on_minutes=int(dec_block.get("min_on_minutes", 30)),
            min_off_minutes=int(dec_block.get("min_off_minutes", 30)),
            max_daily_lighting_hours=float(dec_block.get("max_daily_lighting_hours", 8.0)),
            max_photoperiod_hours=float(dec_block.get("max_photoperiod_hours", 16.0)),
            allowed_windows=_parse_windows(dec_block.get("allowed_windows", [])),
            urgency_margin_hours=float(dec_block.get("urgency_margin_hours", 0.5)),
        ),
        forecast=ForecastConfig(
            mode=fc_block.get("mode", "standard"),
            mode_factors={k: float(v) for k, v in
                          (fc_block.get("mode_factors", {}) or {}).items()} or
                         {"conservative": 0.75, "standard": 1.0, "aggressive": 1.25},
            clear_sky_quantile=float(fc_block.get("clear_sky_quantile", 0.9)),
            doy_window_days=int(fc_block.get("doy_window_days", 15)),
            min_clearness_samples=int(fc_block.get("min_clearness_samples", 6)),
            min_reference_dli=float(fc_block.get("min_reference_dli", 0.5)),
        ),
        tariff=_parse_tariff(tariff_block),
        logger_dir=root / data_block.get("logger_dir", "data/logger"),
        external_dir=root / data_block.get("external_dir", "data/external"),
        unverified_groups=unverified,
        warnings=warnings,
    )
    _validate(cfg)
    return cfg
