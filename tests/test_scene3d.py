"""3D 시각화 테스트.

시각화라도 **태양 위치는 물리적으로 맞아야** 한다. 해가 엉뚱한 곳에 있으면
"이 도구는 대충 만들었다"는 인상을 주고, 그림 전체의 신뢰가 깨진다.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from src.decision import DisplayStatus
from src.scene3d import (STATUS_COLORS, STATUS_ICONS, ZoneView, build_scene,
                         solar_position, zone_view_from)

SEOUL_LAT, SEOUL_LON = 37.5, 127.0


def _view(zone_id, name, status=DisplayStatus.NOT_NEEDED, lamp_on=False,
          dli=6.0, target=12.0) -> ZoneView:
    return ZoneView(zone_id=zone_id, name=name, treatment="NI",
                    greenhouse_id="gh1", status=status, lamp_on=lamp_on,
                    ppfd=250.0, natural_ppfd=250.0, dli_today=dli,
                    target_dli=target, deficit=max(target - dli, 0), temperature=22.0)


class TestSolarPosition:
    def test_춘분_남중고도는_90에서_위도를_뺀_값(self):
        """춘분(3/20) 정오 남중고도 ≈ 90 − 위도. 천문학적으로 확정된 값이다."""
        el, _ = solar_position(SEOUL_LAT, SEOUL_LON, datetime(2026, 3, 20, 12, 30))
        assert el == pytest.approx(90 - SEOUL_LAT, abs=2.0)

    def test_하지_남중고도가_가장_높다(self):
        summer, _ = solar_position(SEOUL_LAT, SEOUL_LON, datetime(2026, 6, 21, 12, 30))
        winter, _ = solar_position(SEOUL_LAT, SEOUL_LON, datetime(2026, 12, 21, 12, 30))
        assert summer == pytest.approx(90 - SEOUL_LAT + 23.45, abs=2.5)
        assert winter == pytest.approx(90 - SEOUL_LAT - 23.45, abs=2.5)
        assert summer > winter

    def test_한밤중에는_지평선_아래(self):
        el, _ = solar_position(SEOUL_LAT, SEOUL_LON, datetime(2026, 5, 20, 0, 0))
        assert el < 0

    def test_방위각이_동에서_서로_이동한다(self):
        """오전은 동쪽(<180°), 정오 무렵 남쪽, 오후는 서쪽(>180°)."""
        morning = solar_position(SEOUL_LAT, SEOUL_LON, datetime(2026, 5, 20, 8, 0))[1]
        afternoon = solar_position(SEOUL_LAT, SEOUL_LON, datetime(2026, 5, 20, 16, 0))[1]
        assert morning < 180 < afternoon

    def test_고도가_정오에_최대(self):
        elevations = [solar_position(SEOUL_LAT, SEOUL_LON,
                                     datetime(2026, 5, 20, h, 0))[0]
                      for h in range(4, 21)]
        assert elevations.index(max(elevations)) == 12 - 4      # 12시 근처

    def test_위도가_높을수록_남중고도가_낮다(self):
        low, _ = solar_position(33.0, 126.5, datetime(2026, 5, 20, 12, 30))
        high, _ = solar_position(41.0, 126.5, datetime(2026, 5, 20, 12, 30))
        assert low > high


class TestSceneBuilding:
    def test_씬이_만들어진다(self, cfg):
        views = {z.id: _view(z.id, z.name) for z in cfg.zones}
        fig = build_scene(cfg, views, datetime(2026, 5, 20, 14, 0), external_ppfd=800.0)
        assert len(fig.data) > 10

    def test_카메라_유지_설정이_있다(self, cfg):
        """uirevision 이 없으면 5초마다 다시 그릴 때 회전·확대가 초기화돼 볼 수가 없다."""
        views = {z.id: _view(z.id, z.name) for z in cfg.zones}
        fig = build_scene(cfg, views, datetime(2026, 5, 20, 14, 0))
        assert fig.layout.uirevision

    def test_점등하면_요소가_늘어난다(self, cfg):
        """등기구 글로우와 광 원뿔이 추가되므로."""
        off = {z.id: _view(z.id, z.name, lamp_on=False) for z in cfg.zones}
        on = {z.id: _view(z.id, z.name, lamp_on=True) for z in cfg.zones}
        n_off = len(build_scene(cfg, off, datetime(2026, 5, 20, 14, 0)).data)
        n_on = len(build_scene(cfg, on, datetime(2026, 5, 20, 14, 0)).data)
        assert n_on > n_off

    def test_광원뿔을_끌_수_있다(self, cfg):
        on = {z.id: _view(z.id, z.name, lamp_on=True) for z in cfg.zones}
        with_cones = len(build_scene(cfg, on, datetime(2026, 5, 20, 14, 0)).data)
        without = len(build_scene(cfg, on, datetime(2026, 5, 20, 14, 0),
                                  show_light_cones=False).data)
        assert with_cones > without

    def test_온실_하나만_그릴_수_있다(self, cfg):
        views = {z.id: _view(z.id, z.name) for z in cfg.zones}
        full = len(build_scene(cfg, views, datetime(2026, 5, 20, 14, 0)).data)
        one = len(build_scene(cfg, views, datetime(2026, 5, 20, 14, 0),
                              only_greenhouse=cfg.greenhouses[0].id).data)
        assert one <= full

    def test_알수없는_온실_id면_전체를_그린다(self, cfg):
        views = {z.id: _view(z.id, z.name) for z in cfg.zones}
        fig = build_scene(cfg, views, datetime(2026, 5, 20, 14, 0),
                          only_greenhouse="없는온실")
        assert len(fig.data) > 10

    def test_상태별로_바닥색이_다르다(self, cfg):
        colors = set()
        for status in DisplayStatus:
            views = {z.id: _view(z.id, z.name, status=status) for z in cfg.zones}
            fig = build_scene(cfg, views, datetime(2026, 5, 20, 14, 0))
            colors |= {tr.color for tr in fig.data
                       if getattr(tr, "color", None) == STATUS_COLORS[status]}
        assert len(colors) == len(DisplayStatus)

    def test_일몰_후에도_그려진다(self, cfg):
        views = {z.id: _view(z.id, z.name) for z in cfg.zones}
        fig = build_scene(cfg, views, datetime(2026, 5, 20, 23, 0), external_ppfd=0.0)
        assert len(fig.data) > 10

    def test_구역_상태가_없어도_깨지지_않는다(self, cfg):
        """업로드 데이터로 구역이 바뀐 직후 등 뷰가 비어 있는 순간이 있다."""
        fig = build_scene(cfg, {}, datetime(2026, 5, 20, 14, 0))
        assert len(fig.data) > 5

    def test_외부광량이_결측이어도_동작한다(self, cfg):
        views = {z.id: _view(z.id, z.name) for z in cfg.zones}
        assert build_scene(cfg, views, datetime(2026, 5, 20, 14, 0),
                           external_ppfd=float("nan")).data


class TestZoneView:
    def test_판단결과에서_뷰를_만든다(self, cfg, params):
        from src.decision import decide
        from tests.conftest import make_state
        state = make_state(now=datetime(2026, 5, 15, 20, 0))
        v = zone_view_from(cfg, state, decide(state, params))
        assert v.zone_id == state.zone_id
        assert v.target_dli == cfg.decision.target_dli
        assert v.status in DisplayStatus

    def test_달성률_계산(self):
        assert _view("z", "z", dli=6.0, target=12.0).progress == pytest.approx(0.5)
        assert _view("z", "z", dli=15.0, target=12.0).progress == pytest.approx(1.25)

    def test_목표가_0이면_0으로_처리(self):
        assert _view("z", "z", dli=5.0, target=0.0).progress == 0.0

    def test_모든_상태에_색과_아이콘이_있다(self):
        for status in DisplayStatus:
            assert status in STATUS_COLORS
            assert status in STATUS_ICONS
