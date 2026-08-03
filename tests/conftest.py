"""테스트 공용 픽스처."""

from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (DecisionConfig, ExternalConfig, ForecastConfig, GreenhouseConfig,
                        LampConfig, NIConfig, SiteConfig, TariffConfig, TariffSeason,
                        TariffSlot, ZoneConfig)
from src.decision import DecisionParams, DecisionState
from src.ingest import SiteFrame
from src.tariff import TariffLookup
from src.timeutil import TimeWindow


def make_zone(zone_id="z1", treatment="NI", reference=True, gh="gh1") -> ZoneConfig:
    return ZoneConfig(id=zone_id, name=zone_id, treatment=treatment, greenhouse_id=gh,
                      greenhouse_name=gh, reference=reference,
                      logger_id=f"logger-{zone_id}", ppfd_column="PPFD")


@pytest.fixture
def cfg() -> SiteConfig:
    """테스트용 최소 설정. 값은 계산이 손으로 검산 가능하도록 딱 떨어지게 잡았다."""
    zones = [make_zone("z1", "NI", True), make_zone("z2", "NI+SL", False)]
    return SiteConfig(
        name="test", crop="test", timezone="Asia/Seoul", interval_minutes=10,
        greenhouses=[GreenhouseConfig(id="gh1", name="gh1", zones=zones)],
        external=ExternalConfig(ppfd_column="ext", night_threshold=2.0,
                                daylight_threshold=50.0),
        # 100 µmol 을 1시간 → 0.36 mol. 검산이 쉬운 값.
        lamp=LampConfig(power_w_per_fixture=100.0, fixtures_per_zone=10,
                        ppfd_contribution=100.0),
        ni=NIConfig(enabled=True,
                    window=TimeWindow(time(18, 30), time(2, 30), True, True, "NI"),
                    treatments=["NI", "NI+SL"]),
        decision=DecisionConfig(
            target_dli=12.0, on_threshold=150.0, off_threshold=250.0,
            moving_average_minutes=30, min_on_minutes=30, min_off_minutes=30,
            max_daily_lighting_hours=8.0, max_photoperiod_hours=16.0,
            allowed_windows=[TimeWindow.parse("04:00-08:00"),
                             TimeWindow.parse("15:00-22:00")],
            urgency_margin_hours=0.5),
        forecast=ForecastConfig(mode="standard",
                                mode_factors={"conservative": 0.75, "standard": 1.0,
                                              "aggressive": 1.25},
                                clear_sky_quantile=0.9, doy_window_days=15,
                                min_clearness_samples=6, min_reference_dli=0.5),
        tariff=TariffConfig(currency="KRW", seasons=[TariffSeason(
            name="all", months=list(range(1, 13)),
            slots=[TariffSlot("경부하", TimeWindow.parse("23:00-09:00"), 60.0),
                   TariffSlot("중간부하", TimeWindow.parse("09:00-17:00"), 90.0),
                   TariffSlot("최대부하", TimeWindow.parse("17:00-23:00"), 130.0)])]),
        logger_dir=ROOT / "data" / "logger", external_dir=ROOT / "data" / "external",
    )


@pytest.fixture
def params(cfg) -> DecisionParams:
    return DecisionParams(decision=cfg.decision, lamp=cfg.lamp, ni=cfg.ni,
                          interval_minutes=cfg.interval_minutes,
                          tariff=TariffLookup(cfg.tariff))


def make_state(**kw) -> DecisionState:
    """기본값: 5월 어느 날 16:00, 소등 상태, 자연광 부족 상황."""
    base = dict(
        now=datetime(2026, 5, 15, 16, 0),
        zone_id="z1", treatment="NI",
        ppfd_ma=100.0, dli_today=5.0, forecast_remaining=1.0,
        lamp_on=False, minutes_in_state=120, lighting_minutes_today=0.0,
        clear_sky_ppfd=600.0, ppfd_now=100.0, ppfd_source="measured",
    )
    base.update(kw)
    return DecisionState(**base)


@pytest.fixture
def sample_frame() -> SiteFrame:
    """3일치 간단한 합성 프레임 (정오 피크의 삼각형 곡선)."""
    idx = pd.date_range("2026-05-01", "2026-05-04", freq="10min", inclusive="left")
    minutes = np.array([t.hour * 60 + t.minute for t in idx])
    shape = np.clip(1 - np.abs(minutes - 750) / 400, 0, None)
    ext = shape * 1500
    ppfd = pd.DataFrame({"z1": shape * 600, "z2": shape * 540}, index=idx)
    return SiteFrame.from_wide(ppfd, pd.Series(ext, index=idx), 10)
