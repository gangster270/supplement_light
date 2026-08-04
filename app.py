"""보광 의사결정 서비스 — Streamlit 시범 서비스.

★ 이 파일에는 **판단 로직이 없다.** 전부 src/decision.py(순수 함수)와 src/engine.py 를
   호출하기만 한다. 화면과 백테스트가 서로 다른 규칙으로 굴러가면 시연에서 본 것이
   현장에서 재현되지 않는다.

역할: "실시간 센서 데이터가 들어오면 이런 원리로 보광 여부를 판단하고,
      향후 실제 제어기와 연결할 수 있다"를 보여주는 기술 시제품.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.control import build_command, to_json
from src.datasource.replay import ReplayDataSource
from src.decision import DisplayStatus
from src.engine import DecisionEngine, prepare
from src.ingest import (build_frame_from_columns, detect_timestamp_column,
                        guess_external_column, guess_ppfd_columns, guess_temperature_column,
                        numeric_columns,
                        read_table)
from src.logbook import LogBook
from src.metrics import cumulative_dli, fill_missing, moving_average
from src.scenario import (DEFAULT_SCENARIO, SCENARIOS, build_scenario_frame,
                          build_training_frame, inject_sensor_fault)
from src.scene3d import STATUS_ICONS, build_scene, solar_position, zone_view_from

st.set_page_config(page_title="보광 의사결정 서비스", page_icon="🌱", layout="wide")

# 배너 배경색. 아이콘은 3D 화면과 같은 것을 쓴다 (scene3d.STATUS_ICONS).
STATUS_BG = {
    DisplayStatus.RECOMMEND_ON: ("#1b5e20", "#c8e6c9"),
    DisplayStatus.NOT_NEEDED: ("#37474f", "#eceff1"),
    DisplayStatus.DEFER: ("#e65100", "#ffe0b2"),
    DisplayStatus.CHECK_SENSOR: ("#b71c1c", "#ffcdd2"),
}


# =====================================================================
# 상태 준비
# =====================================================================

@st.cache_resource(show_spinner=False)
def _base_config():
    return load_config()


def _reset_run() -> None:
    """데이터·설정이 바뀌면 판단 상태를 처음부터 다시 시작해야 한다.

    prepared_key/param_key 는 호출부가 직접 관리한다 — 여기서 지우면 다음 실행에서
    또 리셋으로 판정되어 한 번 더 도는 낭비가 생긴다.
    """
    for k in ("source", "engine", "logbook"):
        st.session_state.pop(k, None)


@st.cache_resource(show_spinner="청천 프로파일과 예측기를 학습하는 중…")
def _prepare_cached(cache_key: str, _cfg, _frame, _training):
    """예측기 준비는 무겁다. 데이터·설정이 같으면 재사용한다."""
    if _training is not None:
        profiles, forecasters = prepare(_cfg, _training)
    else:
        profiles, forecasters = prepare(_cfg, _frame)
    return profiles, forecasters


def _status_banner(decision, state) -> None:
    fg, bg = STATUS_BG[decision.status]
    icon = STATUS_ICONS[decision.status]
    st.markdown(
        f"""<div style="background:{bg};color:{fg};padding:18px 22px;border-radius:12px;
        border-left:10px solid {fg};">
        <div style="font-size:13px;opacity:.8;">{state.now:%Y-%m-%d %H:%M} · {decision.layer.value}</div>
        <div style="font-size:34px;font-weight:800;line-height:1.25;">{icon} {decision.status.value}</div>
        <div style="font-size:15px;margin-top:6px;">{decision.reason}</div>
        </div>""",
        unsafe_allow_html=True)
    if decision.warning:
        st.warning(f"⚠ {decision.warning}")
    if decision.blocked_by and decision.status is not DisplayStatus.DEFER:
        st.info(decision.blocked_by)


# =====================================================================
# 사이드바 — 데이터 소스
# =====================================================================

cfg = _base_config()
st.sidebar.title("🌱 보광 의사결정")
st.sidebar.caption("실험 사이트 한정 시범 서비스 · 실제 제어는 하지 않습니다")

mode = st.sidebar.radio("데이터 소스", ["데모 시나리오", "파일 업로드"],
                        help="데모는 합성 모형이고, 업로드는 실제 센서 파일입니다.")

frame = None
training_frame = None
cache_key = ""

if mode == "데모 시나리오":
    scenario_key = st.sidebar.selectbox(
        "자연광 시나리오", list(SCENARIOS),
        index=list(SCENARIOS).index(DEFAULT_SCENARIO),
        format_func=lambda k: SCENARIOS[k].name)
    st.sidebar.caption(SCENARIOS[scenario_key].description)
    demo_day = st.sidebar.date_input("시연 날짜", value=date(2026, 5, 20))
    # 00:00 에서 시작하면 NI 구간이라 "광량 판단"이 아니라 스케줄 점등부터 보게 된다.
    # 시연에서는 자연광 판단이 드러나는 아침부터 시작하는 편이 낫다.
    start_hour = st.sidebar.slider("시작 시각", 0, 23, 6, format="%d시")
    fault = st.sidebar.checkbox("센서 통신 두절 재현 (11:00부터 90분)",
                                help="안전 조건(센서 확인) 동작을 보여줍니다.")
    cache_key = f"demo|{scenario_key}|{demo_day}|{fault}|{start_hour}"
    frame = build_scenario_frame(cfg, scenario_key, demo_day)
    if fault:
        frame = inject_sensor_fault(
            frame, cfg.zone_ids[0], datetime.combine(demo_day, datetime.min.time())
            + timedelta(hours=11), 90)
    training_frame = build_training_frame(cfg, demo_day)

else:
    uploaded = st.sidebar.file_uploader("센서 데이터 (CSV / Excel)", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        st.info("왼쪽에서 센서 파일을 올리거나 '데모 시나리오'를 선택하세요.")
        st.markdown(
            "**지원 형식** — ZL6 export(.xlsx, 3행 헤더 자동 인식) 또는 "
            "`timestamp, B5_5_PPFD, B5_6_PPFD, …` 형태의 CSV·Excel")
        st.stop()

    raw = read_table(uploaded, uploaded.name)
    ts_col = st.sidebar.selectbox(
        "시각 컬럼", list(raw.columns),
        index=list(raw.columns).index(detect_timestamp_column(raw))
        if detect_timestamp_column(raw) in list(raw.columns) else 0)
    num_cols = numeric_columns(raw, ts_col)
    if not num_cols:
        st.error("숫자로 읽을 수 있는 컬럼이 없습니다. 파일 형식을 확인하세요.")
        st.stop()

    candidates = guess_ppfd_columns(num_cols)
    picked = st.sidebar.multiselect("측정 구역(PPFD) 컬럼", num_cols,
                                    default=candidates[:4])
    if not picked:
        st.warning("구역으로 쓸 PPFD 컬럼을 하나 이상 선택하세요.")
        st.stop()

    with st.sidebar.expander("선택 항목", expanded=False):
        temp_guess = guess_temperature_column(num_cols)
        temp_col = st.selectbox("기온 컬럼 (안전 조건용)", ["(없음)"] + num_cols,
                                index=(num_cols.index(temp_guess) + 1) if temp_guess else 0)
        ext_guess = guess_external_column(num_cols)
        ext_col = st.selectbox("외부 PPFD 컬럼", ["(없음)"] + num_cols,
                               index=(num_cols.index(ext_guess) + 1) if ext_guess else 0,
                               help="온실 밖 기준 광량. 있으면 투과율 산출과 결측 보정에 씁니다.")
        farm = st.text_input("농가명", "")
        house = st.text_input("온실명", "업로드 데이터")

    treatments = {}
    with st.sidebar.expander("구역별 처리 지정", expanded=False):
        st.caption("NI(야간중단) 처리구는 NI 시간대에 강제 점등됩니다.")
        for c in picked:
            treatments[c] = st.selectbox(f"{c}", ["무처리", "NI", "NI+SL"], index=0,
                                         key=f"treat_{c}")

    cfg = cfg.with_zones([(c, c, treatments.get(c, "무처리")) for c in picked],
                         greenhouse_name=house or "업로드 데이터")
    frame = build_frame_from_columns(
        raw, ts_col, {c: c for c in picked}, cfg.interval_minutes,
        external_column=None if ext_col == "(없음)" else ext_col,
        temperature_column=None if temp_col == "(없음)" else temp_col)
    frame = fill_missing(frame, cfg) if frame.external.notna().any() else frame
    cache_key = f"upload|{uploaded.name}|{picked}|{ts_col}"

# 데이터/설정이 바뀌면 진행 중이던 판단을 리셋한다
if st.session_state.get("prepared_key") != cache_key:
    _reset_run()
    st.session_state["prepared_key"] = cache_key


# =====================================================================
# 사이드바 — 연구 파라미터
# =====================================================================

st.sidebar.divider()
st.sidebar.subheader("연구 파라미터")
st.sidebar.caption("실험 결과에 따라 바뀌는 값입니다. 기본값은 임시값입니다.")

from dataclasses import replace as _replace  # noqa: E402

target = st.sidebar.number_input("목표 DLI (mol·m⁻²·d⁻¹)", 1.0, 40.0,
                                 float(cfg.decision.target_dli), 0.5)
lamp_ppfd = st.sidebar.number_input("등기구 점등 시 구역 PPFD (µmol·m⁻²·s⁻¹)",
                                    1.0, 800.0, float(cfg.lamp.ppfd_contribution), 5.0)
lamp_w = st.sidebar.number_input("등기구 소비전력 (W/대)", 1.0, 2000.0,
                                 float(cfg.lamp.power_w_per_fixture), 10.0)
lamp_n = st.sidebar.number_input("구역당 등기구 수", 1, 500, int(cfg.lamp.fixtures_per_zone))
fc_mode = st.sidebar.select_slider("예측 모드", ["conservative", "standard", "aggressive"],
                                   value=cfg.forecast.mode,
                                   format_func=lambda m: {"conservative": "보수적",
                                                          "standard": "표준",
                                                          "aggressive": "공격적"}[m])
max_temp = st.sidebar.number_input("고온 차단 기준 (℃)", 15.0, 50.0,
                                   float(cfg.safety.max_air_temperature), 0.5)

# 보광 허용 시간대는 '정책'이다. 아침·저녁만 보광할지(광주기 연장형),
# 흐린 날 낮에도 보광할지(DLI 보충형)에 따라 판단이 완전히 달라진다.
WINDOW_PRESETS = {
    "아침·저녁 (04-08, 15-22)": ["04:00-08:00", "15:00-22:00"],
    "주간 포함 (04-22)": ["04:00-22:00"],
    "종일 (00-24)": ["00:00-23:59"],
}
preset = st.sidebar.selectbox(
    "보광 허용 시간대", list(WINDOW_PRESETS), index=0,
    help="이 시간대 밖에서는 (NI 를 제외하고) 점등 권고가 나오지 않습니다.")

from src.timeutil import TimeWindow as _TimeWindow  # noqa: E402

cfg = _replace(
    cfg,
    decision=_replace(cfg.decision, target_dli=target,
                      allowed_windows=[_TimeWindow.parse(w, label=w)
                                       for w in WINDOW_PRESETS[preset]]),
    lamp=_replace(cfg.lamp, ppfd_contribution=lamp_ppfd,
                  power_w_per_fixture=lamp_w, fixtures_per_zone=int(lamp_n)),
    forecast=cfg.forecast.with_mode(fc_mode),
    safety=_replace(cfg.safety, max_air_temperature=max_temp,
                    resume_air_temperature=min(cfg.safety.resume_air_temperature, max_temp)),
)

param_key = f"{target}|{lamp_ppfd}|{lamp_w}|{lamp_n}|{fc_mode}|{max_temp}|{preset}"
if st.session_state.get("param_key") != param_key:
    _reset_run()
    st.session_state["param_key"] = param_key


# =====================================================================
# 엔진 준비
# =====================================================================

# 청천 프로파일은 목표 DLI·고온 기준·예측 모드와 무관하다. 등기구 PPFD 만
# 자연광 복원에 영향을 주므로 그것만 캐시 키에 넣는다 (슬라이더마다 8초씩 재학습하지 않도록).
profiles, forecasters = _prepare_cached(f"{cache_key}|lamp={lamp_ppfd}", cfg, frame,
                                        training_frame)

start_at = frame.index[0].to_pydatetime()
if mode == "데모 시나리오":
    candidate = datetime.combine(demo_day, datetime.min.time()) + timedelta(hours=start_hour)
    if frame.index[0] <= pd.Timestamp(candidate) <= frame.index[-1]:
        start_at = candidate

if "source" not in st.session_state:
    st.session_state["source"] = ReplayDataSource(frame, start=start_at)
    st.session_state["engine"] = DecisionEngine(cfg, profiles, forecasters)
    st.session_state["logbook"] = LogBook()

source: ReplayDataSource = st.session_state["source"]
engine: DecisionEngine = st.session_state["engine"]
logbook: LogBook = st.session_state["logbook"]

zone_id = st.sidebar.selectbox("표시 구역", cfg.zone_ids,
                               format_func=lambda z: cfg.zone(z).name)


# =====================================================================
# 재생 제어
# =====================================================================

st.sidebar.divider()
with st.sidebar.expander("🏠 3D 화면 설정", expanded=False):
    show_3d = st.checkbox("3D 온실 현황 표시", value=True)
    scope_all = st.radio("표시 범위", ["선택 온실만", "전체 온실"], index=0,
                         horizontal=True,
                         help="온실이 여러 동이면 '선택 온실만'이 읽기 쉽습니다.") == "전체 온실"
    show_cones = st.checkbox("광 분포(원뿔) 표시", value=True)
    st.caption("치수는 config/site.yaml 의 layout 블록에서 옵니다. "
               "시각화 전용이며 판단 결과에는 영향을 주지 않습니다.")

st.sidebar.divider()
st.sidebar.subheader("재생 제어")
speed = st.sidebar.select_slider(
    "속도 (실제 1초당 가상 시간)", [10, 60, 300, 600, 1800],
    value=600, format_func=lambda s: f"{s // 60}분" if s >= 60 else f"{s}초")

c1, c2, c3 = st.sidebar.columns(3)
if c1.button("▶ 시작", width='stretch'):
    source.start_playback(speed)
if c2.button("⏸ 정지", width='stretch'):
    source.pause_playback()
if c3.button("⏭ 1스텝", width='stretch'):
    source.pause_playback()
    source.step()
    engine.step(source)
    s, d = engine.evaluate(source, zone_id)
    logbook.append(s, d, cfg.zone(zone_id).name)

if st.sidebar.button("⟲ 처음으로", width='stretch'):
    _reset_run()
    st.rerun()

if not cfg.is_fully_verified:
    st.sidebar.warning("미확정 파라미터: " + ", ".join(cfg.unverified_groups))


# =====================================================================
# 본문
# =====================================================================

zone = cfg.zone(zone_id)
st.title("보광 의사결정 서비스")
head = f"{zone.greenhouse_name} · {zone.name}"
if zone.treatment:
    head += f" ({zone.treatment})"
st.caption(head + f" · 목표 DLI {cfg.decision.target_dli:.1f} mol·m⁻²·d⁻¹")

if frame.report.get("synthetic"):
    st.caption("⚠ 지금 보고 있는 값은 **시연용 합성 모형**입니다. 실제 관측값이 아닙니다.")


@st.fragment(run_every="5s")
def live_panel() -> None:
    """실시간 영역만 주기적으로 다시 그린다 (화면 전체를 새로 그리지 않는다)."""
    if source.is_playing:
        before = source.now()
        source.sync()
        # 건너뛴 스텝도 빠짐없이 판단한다 — 배속이 높아도 이력에 구멍이 생기면 안 된다
        if source.now() > before:
            cursor_target = source.cursor
            source.seek(before)
            while source.cursor < cursor_target:
                source.step()
                engine.step(source)
                s_, d_ = engine.evaluate(source, zone_id)
                logbook.append(s_, d_, cfg.zone(zone_id).name)

    state, decision = engine.evaluate(source, zone_id)
    _status_banner(decision, state)

    # --- 3D 온실 현황 ---
    all_views = {z: zone_view_from(cfg, *engine.evaluate(source, z)) for z in cfg.zone_ids}
    ext_series = source.external_history()
    ext_now = float(ext_series.iloc[-1]) if len(ext_series) else None

    if show_3d:
        st.markdown("#### 온실 현황 (3D)")
        elevation, azimuth = solar_position(cfg.latitude, cfg.longitude, state.now)
        sun_txt = (f"태양 고도 {elevation:.0f}° · 방위 {azimuth:.0f}°" if elevation > 0
                   else "일몰 후")
        ext_txt = "—" if ext_now is None or np.isnan(ext_now) else f"{ext_now:,.0f} µmol"
        st.caption(f"{state.now:%Y-%m-%d %H:%M} · {sun_txt} · 외부 광량 {ext_txt} "
                   f"— 마우스로 돌려 보고, 구역·등기구에 올리면 상세 수치가 나옵니다.")
        fig3d = build_scene(cfg, all_views, state.now, external_ppfd=ext_now,
                            show_light_cones=show_cones,
                            only_greenhouse=None if scope_all
                            else cfg.greenhouse_of(zone_id).id)
        st.plotly_chart(fig3d, use_container_width=True,
                        config={"displayModeBar": False},
                        key="scene3d")
        st.caption("바닥 색 = 판단 상태 · 기둥 높이 = DLI 달성률(점선이 목표) · "
                   "노란 등기구 = 점등 중 · 해 위치 = 실제 태양 고도·방위")

    # --- 구역 상태판 (전 구역 한눈에) ---
    st.markdown("#### 구역 상태")
    cols = st.columns(len(all_views))
    for col, (zid, v) in zip(cols, all_views.items()):
        with col:
            st.markdown(f"**{STATUS_ICONS[v.status]} {v.name}**")
            st.caption(f"{v.status.value} · 등기구 {'ON' if v.lamp_on else 'OFF'}")
            st.progress(min(v.progress, 1.0),
                        text=f"DLI {v.dli_today:.1f}/{v.target_dli:.0f} ({v.progress:.0%})")
            ppfd_txt = "—" if v.ppfd is None else f"{v.ppfd:,.0f}"
            temp_txt = "" if v.temperature is None else f" · {v.temperature:.1f}℃"
            st.caption(f"PPFD {ppfd_txt} µmol{temp_txt}")

    st.divider()

    # --- 근거 카드 ---
    st.markdown("#### 판단 근거")
    k1, k2, k3, k4 = st.columns(4)
    ppfd_txt = "—" if state.ppfd_ma is None else f"{state.ppfd_ma:,.0f}"
    k1.metric("현재 PPFD (30분 평균)", f"{ppfd_txt} µmol",
              help="등기구 기여를 포함한 실측 이동평균입니다.")
    if decision.natural_ppfd_estimate is not None:
        k1.caption(f"자연광만: {decision.natural_ppfd_estimate:,.0f} µmol")
    k2.metric("오늘 누적 DLI", f"{state.dli_today:.2f} mol",
              delta=f"목표까지 {max(cfg.decision.target_dli - state.dli_today, 0):.2f}")
    band = ""
    if state.forecast_low is not None and state.forecast_high is not None:
        band = f"{state.forecast_low:.1f} ~ {state.forecast_high:.1f}"
    k3.metric("잔여 자연광 예상", f"{state.forecast_remaining:.2f} mol", help=state.forecast_note)
    if band:
        k3.caption(f"예측 밴드: {band} mol")
    k4.metric("부족 DLI", f"{decision.deficit_dli:.2f} mol",
              delta=f"필요 점등 {decision.hours_needed:.1f}h" if decision.deficit_dli > 0 else "충족")
    if state.forecast_confidence == "low":
        # 해뜨기 전에는 오늘이 맑을지 흐릴지 알 수 없다. '충족'이 확정처럼 보이면 안 된다.
        k4.caption("⚠ 예측 불확실 — 오늘의 흐림 정도를 아직 판단할 수 없습니다")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("예상 최종 DLI", f"{decision.projected_dli:.2f} mol",
              help="현재 누적 + 잔여 자연광 예상 + NI 계획 점등 기여")
    m2.metric("기온", "—" if state.air_temperature is None else f"{state.air_temperature:.1f} ℃",
              help=f"고온 차단 기준 {cfg.safety.max_air_temperature:.1f}℃")
    m3.metric("오늘 점등시간 (NI 포함)", f"{state.lighting_minutes_today / 60:.1f} h",
              help=f"재량 보광 상한은 {cfg.decision.max_daily_lighting_hours:.1f} h 입니다. "
                   f"NI 는 실험 처리 조건이라 상한과 무관하게 점등되므로, 총 점등시간은 "
                   f"상한을 넘을 수 있습니다.")
    kwh = state.lighting_minutes_today / 60 * cfg.lamp.power_kw_per_zone
    m4.metric("오늘 전력량", f"{kwh:.1f} kWh",
              help=f"구역당 {cfg.lamp.power_kw_per_zone:.2f} kW 기준")

    if state.ppfd_source != "measured":
        st.caption(f"※ 현재 값의 출처: **{state.ppfd_source}** — 추정값에 기반한 판단입니다.")

    # --- 그래프 ---
    today = source.today(zone_id)
    if len(today) > 1:
        natural = engine.natural_series(zone_id, today)
        chart = pd.DataFrame({
            "실측 PPFD": today,
            "자연광 추정": natural,
            "30분 이동평균": moving_average(natural, cfg.decision.moving_average_minutes,
                                        cfg.interval_minutes),
        })
        st.markdown("#### 오늘의 광량")
        st.line_chart(chart, height=260)

        nat_cum = cumulative_dli(natural, source.interval_seconds)
        lamp_cum = engine.lamp_dli_series(zone_id, today.index)
        dli_chart = pd.DataFrame({
            "자연광 누적": nat_cum,
            "합계 (자연광 + 보광)": nat_cum + lamp_cum,
            "목표": cfg.decision.target_dli,
        })
        st.markdown("#### 누적 DLI vs 목표")
        st.line_chart(dli_chart, height=240)
        st.caption(f"자연광 {nat_cum.iloc[-1]:.2f} mol + 보광 {lamp_cum.iloc[-1]:.2f} mol "
                   f"= {nat_cum.iloc[-1] + lamp_cum.iloc[-1]:.2f} mol "
                   f"(목표 {cfg.decision.target_dli:.1f} mol)")

    st.caption(f"가상 현재시각 {state.now:%Y-%m-%d %H:%M} · "
               f"{'재생 중' if source.is_playing else '정지'} · 판단 이력 {len(logbook)}건")


live_panel()

st.divider()
tab_cmd, tab_log, tab_help = st.tabs(["제어 명령 미리보기", "판단 이력", "판단 원리"])

with tab_cmd:
    state, decision = engine.evaluate(source, zone_id)
    st.markdown(
        "실제 제어기(PLC·복합환경제어기)와 연결한다면 아래 형태의 명령이 나갑니다. "
        "**현재는 전송하지 않습니다** (`dry_run: true`).")
    st.code(to_json(build_command(state, decision, zone.name, zone.greenhouse_name)),
            language="json")
    st.caption("실제 연동 시에는 워치독, 수동 조작 우선권, 명령 실패 시 폴백을 "
               "함께 설계해야 합니다 — src/control.py 주석 참고.")

with tab_log:
    df = logbook.to_frame()
    if df.empty:
        st.info("아직 판단 이력이 없습니다. 사이드바에서 ▶ 시작 또는 ⏭ 1스텝을 눌러보세요.")
    else:
        summary = logbook.summary(cfg.interval_minutes, cfg.lamp.power_kw_per_zone)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("판단 스텝", f"{summary['steps']:,}")
        s2.metric("점등시간", f"{summary['점등시간']:.1f} h")
        s3.metric("전력량", f"{summary['kWh']:.1f} kWh")
        s4.metric("전기요금", f"{summary['요금']:,.0f} 원")
        st.caption("요금은 계시별 단가를 스텝마다 적용해 합산한 값입니다 "
                   "(기본요금·역률 등 청구서 항목은 제외).")
        dist = pd.DataFrame({
            "판단 스텝 수": pd.Series(summary["상태분포"]),
            "비율": (pd.Series(summary["상태분포"]) / summary["steps"] * 100).round(1),
        })
        c_left, c_right = st.columns(2)
        c_left.markdown("**상태 분포**")
        c_left.dataframe(dist, width='stretch')
        c_right.markdown("**발동 레이어 분포**")
        c_right.dataframe(pd.Series(summary["레이어분포"], name="스텝 수"),
                          width='stretch')
        st.dataframe(df.tail(200), width='stretch', height=320)
        st.download_button("판단 이력 CSV 다운로드", logbook.to_csv_bytes(),
                           file_name=f"판단이력_{zone_id}_{datetime.now():%Y%m%d_%H%M}.csv",
                           mime="text/csv")

with tab_help:
    st.markdown(f"""
### 판단 순서

1. **안전 조건** — 고온({cfg.safety.max_air_temperature:.0f}℃ 이상)이거나 센서가
   {cfg.safety.max_sensor_age_minutes:.0f}분 이상 갱신되지 않으면 광량 판단보다 먼저 걸립니다.
2. **실험 제약** — NI(야간중단) 스케줄 {cfg.ni.window} 은 실험 처리 조건이므로
   광량과 무관하게 점등을 유지합니다. 보광 허용 시간대
   ({', '.join(str(w) for w in cfg.decision.allowed_windows)}) 밖에서는 점등하지 않으며,
   일일 점등상한 {cfg.decision.max_daily_lighting_hours:.0f}시간도 여기서 봅니다.
3. **안정화** — 최소 점등/소등 유지시간 {cfg.decision.min_on_minutes}분, 30분 이동평균,
   점등 임계 {cfg.decision.on_threshold:.0f} / 소등 임계 {cfg.decision.off_threshold:.0f} µmol
   히스테리시스로 구름이 지나갈 때 점·소등이 반복되는 것을 막습니다.
4. **DLI 부족분(주 판단)**

   ```
   부족분 = 목표 DLI − 현재 누적 − 잔여 자연광 예상 − NI 계획 점등 기여
   필요 점등시간 = 부족분 ÷ (등기구 PPFD × 3600 ÷ 1,000,000)
   ```

   남은 가용시간이 필요 점등시간에 근접하면(여유 {cfg.decision.urgency_margin_hours}시간 이하)
   **지금 켜야 한다**고 판단합니다.
5. **저광 임계** — 맑았다면 밝아야 할 시각인데 실제로 어두운 날을 잡습니다.
6. **경제성** — 급하지 않으면 같은 부족분을 계시별 단가가 낮은 시간대에 배치합니다.

### 4가지 상태

| 상태 | 의미 |
|------|------|
| 🟢 보광 권장 | 지금 켜야 오늘 목표를 채웁니다 |
| ⚪ 보광 불필요 | 자연광만으로 충족되거나, 지금은 자연광이 충분합니다 |
| 🟡 보광 보류 | 필요는 하지만 더 유리한 시간대가 있거나 최소 유지시간에 걸렸습니다 |
| 🔴 센서 확인 | 센서·통신 이상. 판단을 멈추고 현재 상태를 유지합니다 |

### 이 서비스가 하지 않는 것

- 실제 등기구를 켜고 끄지 않습니다 (권고만 표시)
- 목표 DLI와 등기구 기여 PPFD는 **연구 파라미터**입니다. 잎들깨 기준값은
  실험 결과로 확정해야 합니다.
- 잔여 자연광 예측은 "남은 시간도 지금과 비슷하게 흐리다"는 가정입니다.
  오전에 맑다가 오후에 흐려지면 빗나갑니다 — 그래서 예측 밴드를 함께 표시합니다.
""")
