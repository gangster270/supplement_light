"""3D 온실 상태 시각화.

★ 이 모듈은 **표현만** 한다. 판단은 전혀 하지 않는다.
   `decide()` 가 내놓은 결과와 측정값을 받아 그림으로 바꿀 뿐이다.

장식이 아니라 **읽는 화면**이 되도록, 3D의 각 요소가 실제 수치를 인코딩한다:

  바닥 타일 색   = 그 구역의 판단 상태 (권장/불필요/보류/센서확인)
  기둥 높이      = 오늘 DLI 달성률 (0~100%+), 목표선이 함께 그려진다
  등기구 밝기    = 실제 점등 신호 (ON이면 발광색 + 광 원뿔)
  해의 위치      = 위경도·시각으로 계산한 실제 태양 고도·방위
  지붕 투명도    = 외부 광량 (밝을수록 투명)

형상 치수는 `config/site.yaml` 의 `layout` 블록에서 온다. 치수가 실제와 달라도
판단 결과는 바뀌지 않는다 (시각화 전용).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import plotly.graph_objects as go

from .config import SiteConfig
from .decision import DisplayStatus

# 상태 색 — 앱의 신호등 배너와 같은 팔레트를 쓴다 (화면 간 색이 달라지면 혼란스럽다)
STATUS_COLORS: dict[DisplayStatus, str] = {
    DisplayStatus.RECOMMEND_ON: "#2e7d32",
    DisplayStatus.NOT_NEEDED: "#78909c",
    DisplayStatus.DEFER: "#ef6c00",
    DisplayStatus.CHECK_SENSOR: "#c62828",
}
STATUS_ICONS: dict[DisplayStatus, str] = {
    DisplayStatus.RECOMMEND_ON: "🟢",
    DisplayStatus.NOT_NEEDED: "⚪",
    DisplayStatus.DEFER: "🟡",
    DisplayStatus.CHECK_SENSOR: "🔴",
}

LAMP_ON_COLOR = "#ffd54f"
LAMP_OFF_COLOR = "#546e7a"
GLASS_COLOR = "#b3e5fc"
FRAME_COLOR = "#78909c"
CANOPY_COLOR = "#1b5e20"


@dataclass
class ZoneView:
    """3D 화면이 필요로 하는 구역 상태 (판단 결과에서 뽑아 온다)."""

    zone_id: str
    name: str
    treatment: str
    greenhouse_id: str
    status: DisplayStatus
    lamp_on: bool
    ppfd: float | None
    natural_ppfd: float | None
    dli_today: float
    target_dli: float
    deficit: float
    temperature: float | None
    reason: str = ""

    @property
    def progress(self) -> float:
        """DLI 달성률 (0~ , 1.0 = 목표 달성)."""
        return self.dli_today / self.target_dli if self.target_dli > 0 else 0.0


# =====================================================================
# 태양 위치
# =====================================================================

def solar_position(latitude: float, longitude: float, moment: datetime,
                   timezone_offset_hours: float = 9.0) -> tuple[float, float]:
    """태양 고도·방위각(도). 표준 천문 근사식(Cooper 적위 + 시간각).

    3D 화면에 해를 놓기 위한 용도라 분 단위 정확도면 충분하다.
    """
    doy = moment.timetuple().tm_yday
    declination = math.radians(23.45 * math.sin(math.radians(360 * (284 + doy) / 365)))
    lat = math.radians(latitude)

    # 균시차 + 경도 보정으로 진태양시를 구한다
    b = math.radians(360 * (doy - 81) / 364)
    eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)   # 분
    standard_meridian = 15.0 * timezone_offset_hours
    clock_hours = moment.hour + moment.minute / 60 + moment.second / 3600
    solar_time = clock_hours + (4 * (longitude - standard_meridian) + eot) / 60
    hour_angle = math.radians(15 * (solar_time - 12))

    sin_elev = (math.sin(lat) * math.sin(declination)
                + math.cos(lat) * math.cos(declination) * math.cos(hour_angle))
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))

    cos_elev = math.cos(math.radians(elevation))
    if abs(cos_elev) < 1e-6:
        return elevation, 180.0
    cos_az = ((math.sin(declination) * math.cos(lat)
               - math.cos(declination) * math.sin(lat) * math.cos(hour_angle)) / cos_elev)
    azimuth = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if hour_angle > 0:                      # 오후에는 서쪽
        azimuth = 360 - azimuth
    return elevation, azimuth


# =====================================================================
# 기본 도형
# =====================================================================

def _box(x0, x1, y0, y1, z0, z1, color, opacity, name, hover=None) -> go.Mesh3d:
    """축에 정렬된 직육면체."""
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 0, 0, 4, 4, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 4, 5, 5, 6, 2, 5, 3, 6, 0, 7]
    k = [2, 3, 5, 4, 6, 7, 6, 6, 7, 7, 4, 4]
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity,
                     name=name, flatshading=True, hoverinfo="text" if hover else "skip",
                     text=hover, showlegend=False)


def _floor_tile(x0, x1, y0, y1, z, color, opacity, name, hover) -> go.Mesh3d:
    return go.Mesh3d(x=[x0, x1, x1, x0], y=[y0, y0, y1, y1], z=[z] * 4,
                     i=[0, 0], j=[1, 2], k=[2, 3], color=color, opacity=opacity,
                     name=name, hoverinfo="text", text=[hover] * 4, showlegend=False)


def _arch_roof(x0, x1, y0, y1, wall_h, ridge_h, color, opacity,
               segments: int = 14) -> go.Mesh3d:
    """반원형(아치) 지붕. 온실 형태를 알아볼 수 있게 하는 요소."""
    ts = np.linspace(0, 1, segments + 1)
    xs = x0 + (x1 - x0) * ts
    zs = wall_h + (ridge_h - wall_h) * np.sin(np.pi * ts)
    X, Y, Z, I, J, K = [], [], [], [], [], []
    for n, (xv, zv) in enumerate(zip(xs, zs)):
        X += [xv, xv]
        Y += [y0, y1]
        Z += [zv, zv]
        if n > 0:
            a, b = 2 * (n - 1), 2 * (n - 1) + 1
            c, d = 2 * n, 2 * n + 1
            I += [a, a]
            J += [b, c]
            K += [c, d]
    return go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, color=color, opacity=opacity,
                     name="지붕", hoverinfo="skip", showlegend=False)


def _light_cone(cx, cy, top_z, radius, color, opacity, segments: int = 16) -> go.Mesh3d:
    """등기구에서 바닥으로 퍼지는 광 원뿔 (점등 중일 때만 그린다)."""
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    X = [cx] + list(cx + radius * np.cos(theta))
    Y = [cy] + list(cy + radius * np.sin(theta))
    Z = [top_z] + [0.02] * segments
    I = [0] * segments
    J = list(range(1, segments + 1))
    K = list(range(2, segments + 1)) + [1]
    return go.Mesh3d(x=X, y=Y, z=Z, i=I, j=J, k=K, color=color, opacity=opacity,
                     name="광 분포", hoverinfo="skip", showlegend=False)


# =====================================================================
# 씬 조립
# =====================================================================

def _zone_bounds(cfg: SiteConfig, gh_index: int, zone_index: int, n_zones: int):
    """구역의 (x0, x1, y0, y1). 온실은 x 방향으로, 구역은 y 방향으로 늘어놓는다."""
    L = cfg.layout
    pitch = L.greenhouse_width_m + L.gap_between_houses_m
    x0 = gh_index * pitch
    zone_len = L.greenhouse_length_m / max(n_zones, 1)
    y0 = zone_index * zone_len
    return x0, x0 + L.greenhouse_width_m, y0, y0 + zone_len


def build_scene(cfg: SiteConfig, views: dict[str, ZoneView], now: datetime,
                external_ppfd: float | None = None,
                show_light_cones: bool = True,
                only_greenhouse: str | None = None) -> go.Figure:
    """온실 3D 상태 화면을 만든다.

    only_greenhouse: 온실 id 를 주면 그 온실만 그린다 (구역이 많을 때 화면이 복잡해지므로).
    """
    L = cfg.layout
    traces: list[go.Mesh3d | go.Scatter3d] = []
    houses = [gh for gh in cfg.greenhouses
              if only_greenhouse is None or gh.id == only_greenhouse]
    if not houses:
        houses = cfg.greenhouses

    # 외부 광량 → 지붕 투명도(밝을수록 투명해 보이게) 와 배경 색
    ext = 0.0 if external_ppfd is None or np.isnan(external_ppfd) else float(external_ppfd)
    brightness = min(ext / 1200.0, 1.0)
    # 너무 투명하면 온실 형태 자체가 안 보인다. 하한을 두고 밝기에 따라 조금만 움직인다.
    roof_opacity = 0.30 - 0.12 * brightness

    label_x, label_y, label_z, label_text = [], [], [], []

    for gi, gh in enumerate(houses):
        n = len(gh.zones)
        gx0, gx1, _, _ = _zone_bounds(cfg, gi, 0, n)

        # --- 온실 외피 ---
        traces.append(_arch_roof(gx0, gx1, 0, L.greenhouse_length_m,
                                 L.wall_height_m, L.ridge_height_m,
                                 GLASS_COLOR, roof_opacity))
        # 아치 리브 — 면만으로는 온실 형태가 안 읽혀서 윤곽선을 함께 그린다
        ts = np.linspace(0, 1, 25)
        arc_x = gx0 + (gx1 - gx0) * ts
        arc_z = L.wall_height_m + (L.ridge_height_m - L.wall_height_m) * np.sin(np.pi * ts)
        for yv in np.linspace(0, L.greenhouse_length_m, n + 1):
            traces.append(go.Scatter3d(
                x=arc_x, y=[yv] * len(ts), z=arc_z, mode="lines",
                line=dict(color=FRAME_COLOR, width=3),
                hoverinfo="skip", showlegend=False))
        traces.append(go.Scatter3d(
            x=[(gx0 + gx1) / 2] * 2, y=[0, L.greenhouse_length_m],
            z=[L.ridge_height_m, L.ridge_height_m], mode="lines",
            line=dict(color=FRAME_COLOR, width=4), hoverinfo="skip", showlegend=False))
        for (a0, a1, b0, b1) in [(gx0, gx0, 0, L.greenhouse_length_m),
                                 (gx1, gx1, 0, L.greenhouse_length_m),
                                 (gx0, gx1, 0, 0),
                                 (gx0, gx1, L.greenhouse_length_m, L.greenhouse_length_m)]:
            traces.append(go.Mesh3d(
                x=[a0, a1, a1, a0], y=[b0, b1, b1, b0],
                z=[0, 0, L.wall_height_m, L.wall_height_m],
                i=[0, 0], j=[1, 2], k=[2, 3], color=GLASS_COLOR,
                opacity=roof_opacity * 0.9, hoverinfo="skip", showlegend=False))

        # 골조(기둥) — 온실처럼 보이게 하는 최소한의 구조물
        for xv in (gx0, gx1):
            for yv in np.linspace(0, L.greenhouse_length_m, n + 1):
                traces.append(_box(xv - 0.06, xv + 0.06, yv - 0.06, yv + 0.06,
                                   0, L.wall_height_m, FRAME_COLOR, 0.85, "골조"))

        # --- 구역별 ---
        for zi, zone in enumerate(gh.zones):
            v = views.get(zone.id)
            x0, x1, y0, y1 = _zone_bounds(cfg, gi, zi, n)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

            if v is None:
                traces.append(_floor_tile(x0, x1, y0, y1, 0.0, "#cfd8dc", 0.6,
                                          zone.name, zone.name))
                continue

            ppfd_txt = "—" if v.ppfd is None else f"{v.ppfd:,.0f}"
            hover = (f"<b>{v.name}</b> ({v.treatment})<br>"
                     f"상태: {STATUS_ICONS[v.status]} {v.status.value}<br>"
                     f"PPFD: {ppfd_txt} µmol<br>"
                     f"누적 DLI: {v.dli_today:.2f} / {v.target_dli:.1f} mol "
                     f"({v.progress:.0%})<br>"
                     f"부족: {v.deficit:.2f} mol<br>"
                     f"등기구: {'점등' if v.lamp_on else '소등'}")
            if v.temperature is not None:
                hover += f"<br>기온: {v.temperature:.1f} ℃"

            # 바닥 타일 = 판단 상태
            traces.append(_floor_tile(x0 + 0.15, x1 - 0.15, y0 + 0.15, y1 - 0.15, 0.01,
                                      STATUS_COLORS[v.status], 0.88, v.name, hover))

            # 작물 군락 — 재배 이미지를 주기 위한 요소
            rows_x = np.linspace(x0 + 1.2, x1 - 1.2, 3)
            rows_y = np.arange(y0 + 0.6, y1 - 0.3, 0.7)
            gx, gy = np.meshgrid(rows_x, rows_y)
            traces.append(go.Scatter3d(
                x=gx.ravel(), y=gy.ravel(),
                z=np.full(gx.size, L.canopy_height_m),
                mode="markers",
                marker=dict(size=3.5, color=CANOPY_COLOR, opacity=0.95,
                            symbol="circle",
                            line=dict(color="#1b5e20", width=1)),
                hoverinfo="skip", showlegend=False, name="작물"))

            # 등기구 + 광 원뿔
            lamp_color = LAMP_ON_COLOR if v.lamp_on else LAMP_OFF_COLOR
            for fx in np.linspace(y0 + (y1 - y0) / (2 * L.fixtures_per_row),
                                  y1 - (y1 - y0) / (2 * L.fixtures_per_row),
                                  L.fixtures_per_row):
                traces.append(_box(cx - 0.9, cx + 0.9, fx - 0.12, fx + 0.12,
                                   L.lamp_height_m, L.lamp_height_m + 0.14,
                                   lamp_color, 1.0, "등기구",
                                   hover=f"{v.name} 등기구 — "
                                         f"{'점등' if v.lamp_on else '소등'}"))
                if v.lamp_on:
                    traces.append(go.Scatter3d(
                        x=[cx], y=[fx], z=[L.lamp_height_m + 0.07], mode="markers",
                        marker=dict(size=11, color=LAMP_ON_COLOR, opacity=0.55,
                                    line=dict(color="#ff8f00", width=1)),
                        hoverinfo="skip", showlegend=False))
                    if show_light_cones:
                        traces.append(_light_cone(cx, fx, L.lamp_height_m, 1.6,
                                                  LAMP_ON_COLOR, 0.16))

            # DLI 달성률 기둥 — 온실 옆에 세우는 3D 막대
            bar_x = gx1 + 0.9
            bar_h = max(v.progress, 0.0) * L.ridge_height_m
            bar_color = ("#2e7d32" if v.progress >= 1 else
                         "#fbc02d" if v.progress >= 0.7 else "#e64a19")
            traces.append(_box(bar_x - 0.28, bar_x + 0.28, cy - 0.9, cy + 0.9,
                               0, max(bar_h, 0.02), bar_color, 0.95, "DLI 달성률",
                               hover=f"{v.name}<br>DLI {v.dli_today:.2f} / "
                                     f"{v.target_dli:.1f} mol ({v.progress:.0%})"))

            # 3D 텍스트는 카메라 각도에 따라 반드시 겹친다. 짧은 식별자만 띄우고
            # 상세 수치는 hover 와 아래 구역 상태표에서 본다.
            short = v.name.split("(")[0].strip()
            label_x.append(bar_x + 0.9)
            label_y.append(cy)
            label_z.append(max(bar_h, 0.02) + 0.5)
            label_text.append(f"{STATUS_ICONS[v.status]} {short} {v.progress:.0%}")

        # 목표선 (달성률 100% 높이) — 기둥이 어디까지 차야 하는지
        traces.append(go.Scatter3d(
            x=[gx1 + 0.9, gx1 + 0.9], y=[-0.5, L.greenhouse_length_m + 0.5],
            z=[L.ridge_height_m, L.ridge_height_m], mode="lines",
            line=dict(color="#455a64", width=3, dash="dash"),
            hoverinfo="skip", showlegend=False, name="목표 DLI"))

    # --- 구역 라벨 ---
    if label_text:
        traces.append(go.Scatter3d(
            x=label_x, y=label_y, z=label_z, mode="text", text=label_text,
            textfont=dict(size=11, color="#263238"), hoverinfo="skip",
            showlegend=False))

    # --- 태양 ---
    elevation, azimuth = solar_position(cfg.latitude, cfg.longitude, now)
    site_w = len(houses) * (L.greenhouse_width_m + L.gap_between_houses_m)
    center_x, center_y = site_w / 2, L.greenhouse_length_m / 2
    radius = max(site_w, L.greenhouse_length_m) * 0.42
    if elevation > 0:
        az = math.radians(azimuth)
        el = math.radians(elevation)
        sx = center_x + radius * math.cos(el) * math.sin(az)
        sy = center_y + radius * math.cos(el) * math.cos(az)
        sz = radius * math.sin(el) + L.ridge_height_m
        traces.append(go.Scatter3d(
            x=[sx], y=[sy], z=[sz], mode="markers+text",
            marker=dict(size=18, color="#ffb300", opacity=0.95,
                        line=dict(color="#ff6f00", width=2)),
            text=[f"☀ 고도 {elevation:.0f}°"], textposition="top center",
            textfont=dict(size=11, color="#e65100"),
            hovertext=f"태양 고도 {elevation:.1f}° / 방위 {azimuth:.0f}°",
            hoverinfo="text", showlegend=False, name="태양"))
        # 햇빛 방향
        traces.append(go.Scatter3d(
            x=[sx, center_x], y=[sy, center_y], z=[sz, L.ridge_height_m],
            mode="lines", line=dict(color="#ffca28", width=4),
            hoverinfo="skip", showlegend=False))
    else:
        traces.append(go.Scatter3d(
            x=[center_x], y=[center_y], z=[radius * 0.5 + L.ridge_height_m],
            mode="markers+text", marker=dict(size=13, color="#5c6bc0", opacity=0.9),
            text=["🌙 일몰 후"], textposition="top center",
            textfont=dict(size=11, color="#3949ab"),
            hoverinfo="text", hovertext=f"태양 고도 {elevation:.1f}° (지평선 아래)",
            showlegend=False))

    # 지면 — 온실이 공중에 떠 보이지 않게
    pad = 2.0
    traces.insert(0, _floor_tile(-pad, site_w + pad, -pad, L.greenhouse_length_m + pad,
                                 -0.02, "#e8eef1", 0.9, "지면", "지면"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="", showticklabels=False, showgrid=False, zeroline=False,
                       showbackground=False),
            yaxis=dict(title="", showticklabels=False, showgrid=False, zeroline=False,
                       showbackground=False),
            zaxis=dict(title="", showticklabels=False, showgrid=False, zeroline=False,
                       showbackground=True,
                       backgroundcolor="rgba(236,242,246,0.55)"),
            aspectmode="data",
            camera=dict(eye=dict(x=0.82, y=-0.98, z=0.55),
                        up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        # ★ 카메라 유지: 5초마다 다시 그려도 회전·확대 상태가 초기화되지 않는다.
        #   이게 없으면 화면을 돌려 볼 수가 없다.
        uirevision="greenhouse-scene",
    )
    return fig


def zone_view_from(cfg: SiteConfig, state, decision) -> ZoneView:
    """DecisionState + Decision → 3D 화면이 쓰는 ZoneView."""
    zone = cfg.zone(state.zone_id)
    return ZoneView(
        zone_id=state.zone_id,
        name=zone.name,
        treatment=zone.treatment,
        greenhouse_id=zone.greenhouse_id,
        status=decision.status,
        lamp_on=decision.should_be_on,
        ppfd=state.ppfd_ma,
        natural_ppfd=decision.natural_ppfd_estimate,
        dli_today=state.dli_today,
        target_dli=cfg.decision.target_dli,
        deficit=decision.deficit_dli,
        temperature=state.air_temperature,
        reason=decision.reason,
    )
