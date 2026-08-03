"""P4 판단 엔진 경계 조건 테스트.

여기서 잡으려는 것은 "대충 맞는 판단"이 아니라 **경계에서 뒤집히는 지점**이다.
임계값 근처, NI 경계, 최소 유지시간 경계, 상한 도달 순간.
"""

from __future__ import annotations

from datetime import datetime, time

import pytest

from src.decision import Action, Layer, Signal, decide
from src.timeutil import TimeWindow
from tests.conftest import make_state


# =====================================================================
# L0: 실험 제약
# =====================================================================

class TestLayer0:
    def test_NI_구간이면_밝아도_점등(self, params):
        """NI는 실험 처리 조건이다. 광량이 아무리 높아도 엔진이 끄면 안 된다."""
        state = make_state(now=datetime(2026, 5, 15, 20, 0), ppfd_ma=2000.0,
                           dli_today=30.0, forecast_remaining=0.0)
        d = decide(state, params)
        assert d.signal is Signal.ON
        assert d.layer is Layer.NI

    def test_NI_경계_포함_여부를_지킨다(self, params):
        """include_start/include_end 설정이 실제로 반영되는지."""
        assert decide(make_state(now=datetime(2026, 5, 15, 18, 30)), params).layer is Layer.NI
        assert decide(make_state(now=datetime(2026, 5, 16, 2, 30)), params).layer is Layer.NI
        # 경계 밖
        assert decide(make_state(now=datetime(2026, 5, 15, 18, 20)), params).layer is not Layer.NI
        assert decide(make_state(now=datetime(2026, 5, 16, 2, 40)), params).layer is not Layer.NI

    def test_NI_구간은_최소유지시간을_무시한다(self, params):
        """NI 시작 시각에는 방금 소등했더라도 즉시 켜야 한다."""
        state = make_state(now=datetime(2026, 5, 15, 18, 30), lamp_on=False,
                           minutes_in_state=0)
        d = decide(state, params)
        assert d.action is Action.TURN_ON
        assert d.blocked_by is None

    def test_NI_대상_처리구가_아니면_강제점등_없음(self, params):
        state = make_state(now=datetime(2026, 5, 15, 20, 0), treatment="무처리",
                           ppfd_ma=2000.0)
        assert decide(state, params).layer is not Layer.NI

    def test_일일_점등상한_도달시_소등(self, params):
        state = make_state(now=datetime(2026, 5, 15, 16, 0),
                           lighting_minutes_today=8 * 60, lamp_on=True,
                           minutes_in_state=10)
        d = decide(state, params)
        assert d.signal is Signal.OFF
        assert d.layer is Layer.DAILY_CAP
        assert d.action is Action.TURN_OFF   # 상한은 최소 유지시간보다 우선

    def test_상한_직전에는_계속_판단한다(self, params):
        """경계 바로 아래(479분)에서는 상한 레이어가 발동하면 안 된다."""
        state = make_state(lighting_minutes_today=8 * 60 - 1)
        assert decide(state, params).layer is not Layer.DAILY_CAP

    def test_허용시간대_밖이면_소등(self, params):
        # 12:00 은 04-08, 15-22 어디에도 없다
        state = make_state(now=datetime(2026, 5, 15, 12, 0), ppfd_ma=0.0)
        d = decide(state, params)
        assert d.signal is Signal.OFF
        assert d.layer is Layer.OUTSIDE_WINDOW

    def test_광주기_상한_도달시_소등(self, params):
        state = make_state(photoperiod_minutes_today=16 * 60)
        d = decide(state, params)
        assert d.layer is Layer.PHOTOPERIOD_CAP
        assert d.signal is Signal.OFF


# =====================================================================
# L1: 안정화 (채터링 방지)
# =====================================================================

class TestLayer1Hold:
    def test_최소_점등유지시간_미달이면_소등을_막는다(self, params):
        """켠 지 10분 만에 밝아졌다고 끄면 등기구 수명과 신뢰를 잃는다."""
        state = make_state(lamp_on=True, minutes_in_state=10, ppfd_ma=2000.0,
                           dli_today=20.0)
        d = decide(state, params)
        assert d.signal is Signal.ON
        assert d.action is Action.KEEP_ON
        assert d.layer is Layer.HOLD
        assert d.blocked_by is not None

    def test_최소_소등유지시간_미달이면_점등을_막는다(self, params):
        state = make_state(lamp_on=False, minutes_in_state=10, ppfd_ma=0.0,
                           dli_today=0.0, forecast_remaining=0.0)
        d = decide(state, params)
        assert d.action is Action.KEEP_OFF
        assert d.layer is Layer.HOLD

    def test_유지시간_충족시_전환된다(self, params):
        state = make_state(lamp_on=False, minutes_in_state=30, ppfd_ma=0.0,
                           dli_today=0.0, forecast_remaining=0.0)
        d = decide(state, params)
        assert d.action is Action.TURN_ON

    def test_구름_통과_시나리오에서_점소등이_반복되지_않는다(self, params):
        """150~250 사이를 오가는 광량에서 상태가 뒤집히면 안 된다 (히스테리시스)."""
        signals = []
        lamp_on, minutes_in_state = True, 60
        for ma in [140, 180, 220, 190, 160, 210, 240, 170]:
            # 점등 중이면 실측에는 등기구 기여 100이 더해져 있다
            state = make_state(lamp_on=lamp_on, minutes_in_state=minutes_in_state,
                               ppfd_ma=ma + (100.0 if lamp_on else 0.0),
                               dli_today=6.0, forecast_remaining=1.0)
            d = decide(state, params)
            if (d.signal is Signal.ON) != lamp_on:
                lamp_on, minutes_in_state = d.signal is Signal.ON, 0
            else:
                minutes_in_state += 10
            signals.append(d.signal)
        # 데드밴드(150~250) 안에서만 움직였으므로 상태 전환이 한 번도 없어야 한다
        assert len(set(signals)) == 1, f"채터링 발생: {signals}"


# =====================================================================
# L2: DLI 부족분
# =====================================================================

class TestLayer2DLI:
    def test_목표_달성_전망이면_소등(self, params):
        state = make_state(dli_today=10.0, forecast_remaining=5.0, ppfd_ma=100.0)
        d = decide(state, params)
        assert d.signal is Signal.OFF
        assert d.layer is Layer.DLI_SUFFICIENT
        assert d.deficit_dli == 0.0

    def test_부족분_계산이_정확하다(self, params):
        """목표12 - 현재5 - 예상1 - NI기여 = 부족분. 손으로 검산 가능해야 한다."""
        state = make_state(now=datetime(2026, 5, 15, 16, 0), dli_today=5.0,
                           forecast_remaining=1.0)
        d = decide(state, params)
        # 16:00~24:00 중 NI 구간은 18:30~24:00 = 5.5시간 → 100µmol × 5.5h = 1.98 mol
        assert d.planned_ni_dli == pytest.approx(1.98, abs=0.01)
        assert d.deficit_dli == pytest.approx(12.0 - 5.0 - 1.0 - 1.98, abs=0.01)
        # 필요 점등시간 = 부족분 / (100µmol → 0.36 mol/h)
        assert d.hours_needed == pytest.approx(d.deficit_dli / 0.36, abs=0.01)

    def test_NI_기여를_빼지_않으면_과잉점등이_된다(self, params):
        """밤에 어차피 켤 불을 낮에 또 켜자고 판단하지 않는지 확인."""
        state = make_state(now=datetime(2026, 5, 15, 16, 0), dli_today=9.0,
                           forecast_remaining=1.5)
        d = decide(state, params)
        # NI 기여 1.98 을 더하면 12.48 로 목표를 넘는다 → 보광 불필요
        assert d.signal is Signal.OFF
        assert d.layer is Layer.DLI_SUFFICIENT

    def test_여유가_없으면_즉시_점등(self, params):
        """남은 가용시간이 필요 점등시간에 근접하면 미루지 않는다."""
        state = make_state(now=datetime(2026, 5, 15, 20, 30), treatment="무처리",
                           dli_today=1.0, forecast_remaining=0.0, ppfd_ma=0.0)
        d = decide(state, params)
        assert d.signal is Signal.ON
        assert d.layer is Layer.DLI_URGENT
        assert d.slack_hours <= params.decision.urgency_margin_hours

    def test_자연광이_충분하면_부족분이_있어도_켜지_않는다(self, params):
        state = make_state(dli_today=2.0, forecast_remaining=0.5, ppfd_ma=400.0)
        d = decide(state, params)
        assert d.deficit_dli > 0
        assert d.signal is Signal.OFF
        assert d.layer is Layer.NATURAL_SUFFICIENT

    def test_점등중_실측에서_등기구_기여를_뺀다(self, params):
        """실측 300 = 자연광 200 + 등기구 100. 자연광은 소등임계 250 미만이므로 유지."""
        state = make_state(lamp_on=True, minutes_in_state=120, ppfd_ma=300.0,
                           dli_today=2.0, forecast_remaining=0.5)
        d = decide(state, params)
        assert d.natural_ppfd_estimate == pytest.approx(200.0)
        assert d.layer is not Layer.NATURAL_SUFFICIENT


# =====================================================================
# L3: 저광 임계
# =====================================================================

class TestLayer3Threshold:
    def test_맑아야_할_시각에_어두우면_점등(self, params):
        """저광일 대응. clear_sky 가 높은데 실측이 낮은 상황."""
        state = make_state(now=datetime(2026, 5, 15, 16, 0), ppfd_ma=80.0,
                           clear_sky_ppfd=600.0, dli_today=6.0, forecast_remaining=1.0)
        d = decide(state, params)
        assert d.signal is Signal.ON
        assert d.layer in (Layer.THRESHOLD, Layer.TARIFF, Layer.DLI_URGENT)

    def test_밤에는_임계만으로_점등하지_않는다(self, params):
        """밤에는 PPFD 가 항상 임계 미만이다. 그걸로 켜면 매일 상한까지 켜진다."""
        state = make_state(now=datetime(2026, 5, 15, 21, 0), treatment="무처리",
                           ppfd_ma=0.0, clear_sky_ppfd=0.0,
                           dli_today=12.5, forecast_remaining=0.0)
        d = decide(state, params)
        assert d.signal is Signal.OFF, "밤에 임계값만으로 점등되면 안 된다"


# =====================================================================
# L4: 경제성
# =====================================================================

class TestLayer4Tariff:
    @staticmethod
    def _params_with_cheap_evening(cfg):
        """앞이 비싸고 뒤가 싼 요금 구조. 연기 판단을 검증하려면 '더 싼 나중'이 있어야 한다."""
        from src.config import TariffConfig, TariffSeason, TariffSlot
        from src.decision import DecisionParams
        from src.tariff import TariffLookup
        tariff = TariffConfig(currency="KRW", seasons=[TariffSeason(
            name="all", months=list(range(1, 13)),
            slots=[TariffSlot("최대부하", TimeWindow.parse("09:00-19:00"), 130.0),
                   TariffSlot("경부하", TimeWindow.parse("19:00-09:00"), 60.0)])])
        return DecisionParams(decision=cfg.decision, lamp=cfg.lamp, ni=cfg.ni,
                              interval_minutes=cfg.interval_minutes,
                              tariff=TariffLookup(tariff))

    def test_여유가_있으면_비싼_시간대를_피한다(self, cfg):
        """15시(최대부하 130원)에 여유가 있으면 19시 이후 경부하(60원)로 미룬다."""
        p = self._params_with_cheap_evening(cfg)
        state = make_state(now=datetime(2026, 5, 15, 15, 0), treatment="무처리",
                           dli_today=11.0, forecast_remaining=0.2, ppfd_ma=50.0,
                           clear_sky_ppfd=50.0)
        d = decide(state, p)
        assert d.slack_hours > p.decision.urgency_margin_hours
        assert d.signal is Signal.OFF
        assert d.layer is Layer.TARIFF_DEFER
        assert d.price_slot == "최대부하"

    def test_싼_시간대가_되면_점등한다(self, cfg):
        """같은 부족분이라도 경부하 구간에 들어오면 켠다."""
        p = self._params_with_cheap_evening(cfg)
        state = make_state(now=datetime(2026, 5, 15, 19, 0), treatment="무처리",
                           dli_today=11.0, forecast_remaining=0.2, ppfd_ma=50.0,
                           clear_sky_ppfd=50.0)
        d = decide(state, p)
        assert d.signal is Signal.ON
        assert d.layer is Layer.TARIFF
        assert d.price_slot == "경부하"

    def test_연기는_당일_안에서만_한다(self, params):
        """DLI 목표는 달력일 단위다. 내일 새벽 경부하로 미루면 오늘 목표를 못 채운다.

        17시 이후 그날 남은 슬롯이 전부 최대부하여도, 더 싼 내일로 미루지 않고 점등한다.
        """
        state = make_state(now=datetime(2026, 5, 15, 17, 0), treatment="무처리",
                           dli_today=11.0, forecast_remaining=0.2, ppfd_ma=50.0,
                           clear_sky_ppfd=50.0)
        d = decide(state, params)
        assert d.signal is Signal.ON
        assert d.layer is Layer.TARIFF

    def test_부족분이_가용시간을_넘으면_긴급(self, params):
        """05시에 부족분 4 mol(=11시간 필요) > 가용 10시간 → 요금 최적화보다 긴급이 우선."""
        state = make_state(now=datetime(2026, 5, 15, 5, 0), treatment="무처리",
                           dli_today=0.0, forecast_remaining=8.0, ppfd_ma=20.0,
                           clear_sky_ppfd=20.0)
        d = decide(state, params)
        assert d.hours_needed > d.hours_available
        assert d.signal is Signal.ON
        assert d.layer is Layer.DLI_URGENT

    def test_달성_불가능한_날은_그렇게_말한다(self, params):
        """켜도 목표에 못 미치는데 '켜면 채워진다'로 읽히면 파라미터 재검토를 놓친다."""
        state = make_state(now=datetime(2026, 5, 15, 20, 0), treatment="무처리",
                           dli_today=0.0, forecast_remaining=0.0, ppfd_ma=0.0)
        d = decide(state, params)
        assert d.hours_needed > d.hours_available
        assert "불가능" in d.reason and "미달" in d.reason

    def test_달성_가능한_긴급은_다르게_말한다(self, params):
        state = make_state(now=datetime(2026, 5, 15, 15, 0), treatment="무처리",
                           dli_today=9.0, forecast_remaining=0.5, ppfd_ma=50.0,
                           clear_sky_ppfd=50.0)
        d = decide(state, params)
        if d.layer is Layer.DLI_URGENT:
            assert d.hours_needed <= d.hours_available
            assert "불가능" not in d.reason

    def test_요금정보가_없어도_동작한다(self, params, cfg):
        from src.config import TariffConfig
        from src.decision import DecisionParams
        p = DecisionParams(decision=cfg.decision, lamp=cfg.lamp, ni=cfg.ni,
                           interval_minutes=10, tariff=None)
        d = decide(make_state(dli_today=2.0, forecast_remaining=0.0, ppfd_ma=10.0), p)
        assert d.signal in (Signal.ON, Signal.OFF)


# =====================================================================
# 근거 / 계약
# =====================================================================

class TestEvidence:
    def test_모든_판단에_근거가_붙는다(self, params):
        """근거 없는 신호는 UI에도 이력에도 쓸 수 없다."""
        cases = [
            make_state(now=datetime(2026, 5, 15, 20, 0)),                    # NI
            make_state(now=datetime(2026, 5, 15, 12, 0)),                    # 허용시간 밖
            make_state(dli_today=20.0, forecast_remaining=0.0),              # 충족
            make_state(dli_today=0.0, forecast_remaining=0.0, ppfd_ma=0.0),  # 부족
        ]
        for state in cases:
            d = decide(state, params)
            assert d.reason and len(d.reason) > 10
            assert d.layer is not None
            assert isinstance(d.evidence, dict)

    def test_action_과_signal_이_일관된다(self, params):
        for lamp_on in (True, False):
            for hour in range(0, 24):
                state = make_state(now=datetime(2026, 5, 15, hour, 0), lamp_on=lamp_on,
                                   minutes_in_state=120)
                d = decide(state, params)
                if d.signal is Signal.ON:
                    assert d.action in (Action.TURN_ON, Action.KEEP_ON)
                    assert (d.action is Action.KEEP_ON) == lamp_on
                else:
                    assert d.action in (Action.TURN_OFF, Action.KEEP_OFF)
                    assert (d.action is Action.KEEP_OFF) == (not lamp_on)

    def test_ppfd_결측이어도_판단이_나온다(self, params):
        """센서가 죽어도 서비스가 멈추면 안 된다 (DLI 기준으로는 판단 가능)."""
        state = make_state(ppfd_ma=None, ppfd_source="missing")
        d = decide(state, params)
        assert d.signal in (Signal.ON, Signal.OFF)
        assert d.natural_ppfd_estimate is None
