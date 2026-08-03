"""판단 추적(trace) CLI — P1~P4가 실제로 연결되어 도는지 확인하는 도구.

지정한 기간을 리플레이하면서 매 스텝 판단을 내고, 판단이 바뀌는 시점과 근거를 출력한다.
정식 백테스트(달성률·전력비·시나리오 비교)는 P5의 backtest.py 에서 다룬다.

사용:
    python3 tools/run_decisions.py --start 2026-05-10 --end 2026-05-13
    python3 tools/run_decisions.py --start 2026-05-10 --end 2026-05-11 --zone gh1_ni --all
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.datasource.replay import ReplayDataSource
from src.engine import DecisionEngine, prepare
from src.ingest import load_site_frame
from src.metrics import estimate_lamp_contribution, fill_missing


def main() -> None:
    ap = argparse.ArgumentParser(description="보광 판단 추적")
    ap.add_argument("--start", required=True, help="추적 시작일 (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="추적 종료일 (제외)")
    ap.add_argument("--zone", default=None, help="구역 id (기본: 첫 구역)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--logger-dir", default=None)
    ap.add_argument("--external-dir", default=None)
    ap.add_argument("--all", action="store_true", help="모든 스텝 출력 (기본: 변화 시점만)")
    ap.add_argument("--target", type=float, default=None,
                    help="목표 DLI 임시 변경 (site.yaml 을 고치지 않고 민감도 확인)")
    ap.add_argument("--mode", default=None,
                    choices=["conservative", "standard", "aggressive"],
                    help="예측 모드 임시 변경")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.target is not None:
        cfg = replace(cfg, decision=replace(cfg.decision, target_dli=args.target))
        print(f"※ 목표 DLI 를 {args.target} 로 임시 변경했습니다 (site.yaml 은 그대로).")
    if args.mode is not None:
        cfg = replace(cfg, forecast=cfg.forecast.with_mode(args.mode))
        print(f"※ 예측 모드를 '{args.mode}' 로 임시 변경했습니다.")
    for w in cfg.warnings:
        print(f"⚠  {w}")
    if not cfg.is_fully_verified:
        print(f"⚠  {cfg.unverified_message()}\n")

    print("데이터 로드 중...")
    frame = load_site_frame(cfg, args.logger_dir, args.external_dir)
    frame = fill_missing(frame, cfg)
    print(f"   {frame.index[0]} ~ {frame.index[-1]}  ({len(frame.index):,}행, "
          f"{len(frame.zone_ids)}구역)")
    lamp_est = estimate_lamp_contribution(frame, cfg)
    print(f"   야간 LED 실측 추정: "
          f"{ {k: round(v, 1) for k, v in lamp_est.items()} }  "
          f"(site.yaml 설정값: {cfg.lamp.ppfd_contribution})")

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    print(f"\n학습 구간: {frame.index[0]} ~ {start} (이후 데이터는 학습에 쓰지 않음)")
    profiles, forecasters = prepare(cfg, frame, training_end=start)
    engine = DecisionEngine(cfg, profiles, forecasters)

    zone_id = args.zone or cfg.zone_ids[0]
    zone = cfg.zone(zone_id)
    print(f"추적 구역: {zone.name} ({zone.treatment}) / 목표 DLI {cfg.decision.target_dli}\n")

    source = ReplayDataSource(frame, start=start.to_pydatetime())
    header = f"{'시각':<17}{'신호':<6}{'PPFD':>7}{'누적DLI':>8}{'예상':>7}{'부족':>7}  발동 레이어"
    print(header)
    print("-" * len(header) * 2)

    prev_signal = None
    rows = []
    for src in source.iter_steps(start.to_pydatetime(), end.to_pydatetime()):
        out = engine.step(src)
        state, decision = out[zone_id]
        rows.append({"시각": state.now, "신호": decision.signal.value,
                     "레이어": decision.layer.value, "부족분": decision.deficit_dli})
        changed = decision.signal != prev_signal
        if args.all or changed:
            mark = "▶" if changed else " "
            ppfd = "-" if state.ppfd_ma is None else f"{state.ppfd_ma:.0f}"
            print(f"{mark}{state.now:%m-%d %H:%M}  {decision.signal.value:<5}"
                  f"{ppfd:>7}{state.dli_today:>8.1f}{state.forecast_remaining:>7.1f}"
                  f"{decision.deficit_dli:>7.1f}  {decision.layer.value}")
            if changed:
                print(f"   └ {decision.reason}")
        prev_signal = decision.signal

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("판단 분포 (레이어별 스텝 수)")
    print(df["레이어"].value_counts().to_string())

    per_day = df[df["신호"] == "ON"].groupby(df["시각"].dt.date).size() * cfg.interval_minutes / 60
    print(f"\n일별 점등시간 (시간)")
    print(per_day.to_string() if len(per_day) else "  (점등 없음)")
    kw = cfg.lamp.power_kw_per_zone
    print(f"\n총 점등 {per_day.sum():.1f}시간 × {kw:.1f}kW = {per_day.sum() * kw:.1f} kWh "
          f"(구역 1개 기준)")
    print("※ 요금 정산과 시나리오 비교는 P5·P7에서 다룹니다.")


if __name__ == "__main__":
    main()
