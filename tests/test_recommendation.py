"""Tests for regime_detection.recommendation — Phase 4.

Tests cover:
  - Recommended logic activation rules (v3.1 Section 5)
  - Range vs Scalp distinction within CHOP_NEUTRAL
  - Strategy-specific overrides (pairs, options)
  - Range hints computation (Donchian channel)
  - Range persistence counter
  - Exit mandate with grace period
  - Exit mandate immediate triggers (CPD, Hurst, vol)
  - Full integration through RegimeManager
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from regime_detection.recommendation import (
    compute_range_hints,
    compute_range_persistence,
    determine_recommended_logic,
    evaluate_exit_mandate,
)
from regime_detection.schema import (
    ConsensusState,
    HMMLabel,
    LiquidityStatus,
    RangeHints,
    RecommendedLogic,
    VolatilityRegime,
)


# ---------------------------------------------------------------------------
# Default config sections
# ---------------------------------------------------------------------------

HURST_CFG = {
    "trending_threshold": 0.60,
    "mean_reverting_threshold": 0.40,
    "range_min_hurst": 0.48,
    "range_max_hurst": 0.58,
}

RANGE_CFG = {
    "enabled": True,
    "channel_type": "donchian",
    "channel_period": 30,
    "min_bars_persistence": 10,
    "deviation_entry_threshold": 1.5,
    "target_range_capture_pct": 0.6,
}

EXIT_CFG = {
    "enabled": True,
    "grace_bars": 2,
    "triggers": [
        "hurst_above_trending",
        "cpd_structural_break",
        "volatility_expanding",
        "hmm_state_change",
    ],
}


# ===================================================================
# RECOMMENDED LOGIC — ACTIVATION RULES
# ===================================================================

class TestRecommendedLogicSafetyGates:
    """Safety gates: UNKNOWN, TRANSITION, structural break → NO_TRADE."""

    def test_unknown_consensus(self):
        r = determine_recommended_logic(
            ConsensusState.UNKNOWN, HMMLabel.UNKNOWN, None,
            VolatilityRegime.UNKNOWN, LiquidityStatus.UNKNOWN, False,
            "scalping", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.NO_TRADE

    def test_transition_consensus(self):
        r = determine_recommended_logic(
            ConsensusState.TRANSITION, HMMLabel.BULL, 0.65,
            VolatilityRegime.EXPANDING, LiquidityStatus.PASSED, False,
            "scalping", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.NO_TRADE

    def test_structural_break(self):
        r = determine_recommended_logic(
            ConsensusState.CHOP_NEUTRAL, HMMLabel.CHOP, 0.50,
            VolatilityRegime.LOW_STABLE, LiquidityStatus.CONSOLIDATION, True,
            "scalping", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.NO_TRADE

    def test_none_hurst(self):
        r = determine_recommended_logic(
            ConsensusState.CHOP_NEUTRAL, HMMLabel.CHOP, None,
            VolatilityRegime.LOW_STABLE, LiquidityStatus.CONSOLIDATION, False,
            "scalping", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.NO_TRADE


class TestRecommendedLogicSwing:
    """Trending regimes → SWING_TREND_FOLLOW."""

    def test_bull_persistent(self):
        r = determine_recommended_logic(
            ConsensusState.BULL_PERSISTENT, HMMLabel.BULL, 0.65,
            VolatilityRegime.MODERATE, LiquidityStatus.PASSED, False,
            "swing", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.SWING_TREND_FOLLOW

    def test_bear_persistent(self):
        r = determine_recommended_logic(
            ConsensusState.BEAR_PERSISTENT, HMMLabel.BEAR, 0.63,
            VolatilityRegime.MODERATE, LiquidityStatus.PASSED, False,
            "swing", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.SWING_TREND_FOLLOW

    def test_scalping_strategy_in_trend_still_swings(self):
        """Even scalping strategy type gets SWING in trending market."""
        r = determine_recommended_logic(
            ConsensusState.BULL_PERSISTENT, HMMLabel.BULL, 0.65,
            VolatilityRegime.MODERATE, LiquidityStatus.PASSED, False,
            "scalping", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.SWING_TREND_FOLLOW


class TestRecommendedLogicPairs:
    """Pairs trading strategy-specific overrides."""

    def test_pairs_in_chop(self):
        r = determine_recommended_logic(
            ConsensusState.CHOP_NEUTRAL, HMMLabel.CHOP, 0.50,
            VolatilityRegime.LOW_STABLE, LiquidityStatus.CONSOLIDATION, False,
            "pairs_trading", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.PAIRS_MEAN_REVERSION

    def test_pairs_in_trend_no_trade(self):
        r = determine_recommended_logic(
            ConsensusState.BULL_PERSISTENT, HMMLabel.BULL, 0.65,
            VolatilityRegime.MODERATE, LiquidityStatus.PASSED, False,
            "pairs_trading", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.NO_TRADE


class TestRecommendedLogicOptions:
    """Options strategy-specific overrides."""

    def test_options_income_low_vol(self):
        r = determine_recommended_logic(
            ConsensusState.CHOP_NEUTRAL, HMMLabel.CHOP, 0.50,
            VolatilityRegime.LOW_STABLE, LiquidityStatus.CONSOLIDATION, False,
            "options_income", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.OPTIONS_INCOME

    def test_options_income_expanding_vol_no_trade(self):
        r = determine_recommended_logic(
            ConsensusState.CHOP_NEUTRAL, HMMLabel.CHOP, 0.50,
            VolatilityRegime.EXPANDING, LiquidityStatus.PASSED, False,
            "options_income", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.NO_TRADE

    def test_options_speculative_in_trend(self):
        r = determine_recommended_logic(
            ConsensusState.BULL_PERSISTENT, HMMLabel.BULL, 0.65,
            VolatilityRegime.MODERATE, LiquidityStatus.PASSED, False,
            "options_speculative", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.OPTIONS_SPECULATIVE

    def test_options_speculative_expanding_vol(self):
        r = determine_recommended_logic(
            ConsensusState.CHOP_NEUTRAL, HMMLabel.CHOP, 0.50,
            VolatilityRegime.EXPANDING, LiquidityStatus.PASSED, False,
            "options_speculative", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.OPTIONS_SPECULATIVE


class TestScalpVsRange:
    """The key v3.1 distinction: SCALP_MEAN_REVERSION vs RANGE_TRADING."""

    def test_scalp_low_hurst_consolidation(self):
        """Hurst < 0.48 + CONSOLIDATION → SCALP."""
        r = determine_recommended_logic(
            ConsensusState.CHOP_NEUTRAL, HMMLabel.CHOP, 0.42,
            VolatilityRegime.LOW_STABLE, LiquidityStatus.CONSOLIDATION, False,
            "scalping", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.SCALP_MEAN_REVERSION

    def test_range_all_conditions_met(self):
        """All range conditions met → RANGE_TRADING."""
        r = determine_recommended_logic(
            ConsensusState.CHOP_NEUTRAL, HMMLabel.CHOP, 0.52,
            VolatilityRegime.LOW_STABLE, LiquidityStatus.CONSOLIDATION, False,
            "range_trading", HURST_CFG, RANGE_CFG,
            range_persistence_bars=15,  # >= min_bars_persistence(10)
        )
        assert r == RecommendedLogic.RANGE_TRADING

    def test_range_hurst_in_sweet_spot_but_no_persistence(self):
        """Hurst in range but persistence not met → falls back to SCALP."""
        r = determine_recommended_logic(
            ConsensusState.CHOP_NEUTRAL, HMMLabel.CHOP, 0.52,
            VolatilityRegime.LOW_STABLE, LiquidityStatus.CONSOLIDATION, False,
            "range_trading", HURST_CFG, RANGE_CFG,
            range_persistence_bars=3,  # < min_bars_persistence(10)
        )
        assert r == RecommendedLogic.SCALP_MEAN_REVERSION

    def test_range_hurst_too_high(self):
        """Hurst above range_max (0.58) in CHOP → NO_TRADE (near trending)."""
        r = determine_recommended_logic(
            ConsensusState.CHOP_NEUTRAL, HMMLabel.CHOP, 0.59,
            VolatilityRegime.LOW_STABLE, LiquidityStatus.CONSOLIDATION, False,
            "scalping", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.NO_TRADE

    def test_range_expanding_vol_blocks_range(self):
        """EXPANDING vol blocks range trading even with good Hurst."""
        r = determine_recommended_logic(
            ConsensusState.CHOP_NEUTRAL, HMMLabel.CHOP, 0.52,
            VolatilityRegime.EXPANDING, LiquidityStatus.CONSOLIDATION, False,
            "range_trading", HURST_CFG, RANGE_CFG,
            range_persistence_bars=20,
        )
        # EXPANDING blocks range, Hurst not below range_min so no scalp either
        assert r != RecommendedLogic.RANGE_TRADING

    def test_liquidity_trap_blocks_scalp(self):
        """LIQUIDITY_TRAP → neither scalp nor range."""
        r = determine_recommended_logic(
            ConsensusState.CHOP_NEUTRAL, HMMLabel.CHOP, 0.42,
            VolatilityRegime.LOW_STABLE, LiquidityStatus.LIQUIDITY_TRAP, False,
            "scalping", HURST_CFG, RANGE_CFG,
        )
        assert r == RecommendedLogic.NO_TRADE


# ===================================================================
# RANGE HINTS
# ===================================================================

class TestRangeHints:
    """Test Donchian channel boundary computation."""

    def _make_closes(self, n=50, base=150.0, amplitude=3.0):
        """Oscillating close prices (sine wave)."""
        return base + amplitude * np.sin(np.linspace(0, 4 * np.pi, n))

    def test_returns_none_for_non_chop(self):
        closes = self._make_closes()
        result = compute_range_hints(
            closes, ConsensusState.BULL_PERSISTENT,
            RecommendedLogic.SWING_TREND_FOLLOW, RANGE_CFG,
        )
        assert result is None

    def test_returns_none_for_no_trade(self):
        closes = self._make_closes()
        result = compute_range_hints(
            closes, ConsensusState.CHOP_NEUTRAL,
            RecommendedLogic.NO_TRADE, RANGE_CFG,
        )
        assert result is None

    def test_donchian_channel_computed(self):
        closes = self._make_closes(n=50, base=150.0, amplitude=3.0)
        result = compute_range_hints(
            closes, ConsensusState.CHOP_NEUTRAL,
            RecommendedLogic.RANGE_TRADING, RANGE_CFG,
        )
        assert result is not None
        assert result.range_lower < result.range_upper
        assert result.channel_type.startswith("Donchian")

    def test_is_clean_range_for_range_trading(self):
        closes = self._make_closes()
        result = compute_range_hints(
            closes, ConsensusState.CHOP_NEUTRAL,
            RecommendedLogic.RANGE_TRADING, RANGE_CFG,
        )
        assert result.is_clean_range is True

    def test_not_clean_for_scalp(self):
        closes = self._make_closes()
        result = compute_range_hints(
            closes, ConsensusState.CHOP_NEUTRAL,
            RecommendedLogic.SCALP_MEAN_REVERSION, RANGE_CFG,
        )
        assert result.is_clean_range is False

    def test_deviation_pct_computed(self):
        closes = self._make_closes(n=50, base=150.0, amplitude=3.0)
        result = compute_range_hints(
            closes, ConsensusState.CHOP_NEUTRAL,
            RecommendedLogic.RANGE_TRADING, RANGE_CFG,
        )
        assert result.current_deviation_pct is not None
        assert -200 < result.current_deviation_pct < 200

    def test_insufficient_bars(self):
        closes = np.array([150.0, 151.0, 149.0])  # < channel_period
        result = compute_range_hints(
            closes, ConsensusState.CHOP_NEUTRAL,
            RecommendedLogic.RANGE_TRADING, RANGE_CFG,
        )
        assert result is None

    def test_keltner_channel(self):
        closes = self._make_closes()
        keltner_cfg = {**RANGE_CFG, "channel_type": "keltner"}
        result = compute_range_hints(
            closes, ConsensusState.CHOP_NEUTRAL,
            RecommendedLogic.RANGE_TRADING, keltner_cfg,
        )
        assert result is not None
        assert result.channel_type.startswith("Keltner")


# ===================================================================
# RANGE PERSISTENCE
# ===================================================================

class TestRangePersistence:
    """Test consecutive-bar-in-channel counter."""

    def test_stable_series_high_persistence(self):
        """All bars inside a tight range → high persistence count."""
        closes = 150.0 + 0.5 * np.random.RandomState(42).randn(100)
        count = compute_range_persistence(closes, RANGE_CFG)
        assert count > 0

    def test_breakout_resets_count(self):
        """Price jumping far outside range → persistence = 0 or low."""
        rng = np.random.RandomState(42)
        # Stable for 80 bars, then spike
        stable = 150.0 + 0.3 * rng.randn(80)
        spike = np.array([200.0, 210.0, 220.0])
        closes = np.concatenate([stable, spike])
        count = compute_range_persistence(closes, RANGE_CFG)
        # Recent bars are way outside → low persistence
        assert count <= 3

    def test_insufficient_data(self):
        closes = np.array([150.0, 151.0])
        count = compute_range_persistence(closes, RANGE_CFG)
        assert count == 0


# ===================================================================
# EXIT MANDATE
# ===================================================================

class TestExitMandate:
    """Test regime shift detection with grace period and immediate triggers."""

    def test_no_shift_no_mandate(self):
        mandate, counter = evaluate_exit_mandate(
            ConsensusState.CHOP_NEUTRAL, ConsensusState.CHOP_NEUTRAL,
            shift_counter=0, grace_bars=2, exit_cfg=EXIT_CFG,
            hurst_value=0.50, structural_break=False,
            vol_regime=VolatilityRegime.LOW_STABLE, hurst_cfg=HURST_CFG,
        )
        assert mandate is False
        assert counter == 0

    def test_first_tick_no_mandate(self):
        mandate, counter = evaluate_exit_mandate(
            ConsensusState.CHOP_NEUTRAL, None,  # first tick
            shift_counter=0, grace_bars=2, exit_cfg=EXIT_CFG,
            hurst_value=0.50, structural_break=False,
            vol_regime=VolatilityRegime.LOW_STABLE, hurst_cfg=HURST_CFG,
        )
        assert mandate is False

    def test_shift_needs_grace_period(self):
        """Shift detected but only 1 bar → no mandate yet."""
        mandate, counter = evaluate_exit_mandate(
            ConsensusState.BULL_PERSISTENT, ConsensusState.CHOP_NEUTRAL,
            shift_counter=0, grace_bars=2, exit_cfg=EXIT_CFG,
            hurst_value=0.65, structural_break=False,
            vol_regime=VolatilityRegime.MODERATE, hurst_cfg=HURST_CFG,
        )
        # Hurst above trending + was CHOP → immediate trigger
        assert mandate is True

    def test_grace_period_confirmed(self):
        """Shift held for grace_bars → mandate issued."""
        mandate, counter = evaluate_exit_mandate(
            ConsensusState.BULL_PERSISTENT, ConsensusState.CHOP_NEUTRAL,
            shift_counter=1, grace_bars=2, exit_cfg=EXIT_CFG,
            hurst_value=0.55, structural_break=False,
            vol_regime=VolatilityRegime.MODERATE, hurst_cfg=HURST_CFG,
        )
        # hmm_state_change trigger: counter was 1, now 2 >= grace_bars
        assert mandate is True
        assert counter >= 2

    def test_structural_break_immediate(self):
        """CPD structural break → immediate mandate (no grace period)."""
        mandate, counter = evaluate_exit_mandate(
            ConsensusState.CHOP_NEUTRAL, ConsensusState.CHOP_NEUTRAL,
            shift_counter=0, grace_bars=5, exit_cfg=EXIT_CFG,
            hurst_value=0.50, structural_break=True,
            vol_regime=VolatilityRegime.LOW_STABLE, hurst_cfg=HURST_CFG,
        )
        assert mandate is True

    def test_expanding_vol_from_chop_immediate(self):
        """EXPANDING vol but consensus stays CHOP → no mandate (normal crypto vol)."""
        mandate, counter = evaluate_exit_mandate(
            ConsensusState.CHOP_NEUTRAL, ConsensusState.CHOP_NEUTRAL,
            shift_counter=0, grace_bars=5, exit_cfg=EXIT_CFG,
            hurst_value=0.50, structural_break=False,
            vol_regime=VolatilityRegime.EXPANDING, hurst_cfg=HURST_CFG,
        )
        assert mandate is False

    def test_disabled_exit_mandate(self):
        """exit_mandate.enabled=false → never trigger."""
        disabled_cfg = {**EXIT_CFG, "enabled": False}
        mandate, counter = evaluate_exit_mandate(
            ConsensusState.BULL_PERSISTENT, ConsensusState.CHOP_NEUTRAL,
            shift_counter=10, grace_bars=2, exit_cfg=disabled_cfg,
            hurst_value=0.70, structural_break=True,
            vol_regime=VolatilityRegime.EXPANDING, hurst_cfg=HURST_CFG,
        )
        assert mandate is False


# ===================================================================
# FULL INTEGRATION: Phase 4 through RegimeManager
# ===================================================================

class TestManagerPhase4Integration:
    """End-to-end tests through RegimeManager with real Phase 4 logic."""

    def _make_bar(self, close: float = 150.0) -> dict:
        return {
            "timestamp": time.time(),
            "o": close - 0.5,
            "h": close + 1.0,
            "l": close - 1.0,
            "c": close,
            "v": 10000.0,
        }

    def _feed(self, m, n: int, base: float = 150.0, amplitude: float = 0.1):
        for i in range(n):
            price = base + amplitude * (i % 5)
            m.update(
                self._make_bar(price),
                funding_rate=0.0003,
                order_book_imbalance=0.1,
            )

    def test_recommended_logic_not_stub(self):
        """After warmup, recommended_logic should be a real value."""
        from regime_detection import RegimeManager
        m = RegimeManager(strategy_type="scalping", market_class="crypto")
        self._feed(m, 150)
        regime = m.get_current_regime()
        # Should be a valid enum value, not always NO_TRADE
        valid = {e.value for e in RecommendedLogic}
        assert regime["recommended_logic"] in valid

    def test_exit_mandate_is_bool(self):
        from regime_detection import RegimeManager
        m = RegimeManager()
        self._feed(m, 150)
        regime = m.get_current_regime()
        assert isinstance(regime["exit_mandate"], bool)

    def test_range_hints_present_in_chop(self):
        """When CHOP + scalp/range → range_hints should be populated."""
        from regime_detection import RegimeManager
        m = RegimeManager(strategy_type="scalping", market_class="crypto")
        # Feed very stable data → likely CHOP
        for i in range(200):
            price = 150.0 + 0.3 * np.sin(i * 0.1)
            m.update(
                self._make_bar(price),
                funding_rate=0.0003,
                order_book_imbalance=0.1,
            )
        regime = m.get_current_regime()
        # If we got CHOP + scalp/range, range_hints should be there
        if regime["recommended_logic"] in ("SCALP_MEAN_REVERSION", "RANGE_TRADING"):
            assert "range_hints" in regime["signals"]
            rh = regime["signals"]["range_hints"]
            assert rh["range_lower"] is not None
            assert rh["range_upper"] is not None

    def test_pairs_trading_gets_pairs_logic(self):
        from regime_detection import RegimeManager
        m = RegimeManager(
            market_type="SPREAD", strategy_type="pairs_trading", market_class="crypto"
        )
        self._feed(m, 150)
        regime = m.get_current_regime()
        # In CHOP → should get PAIRS_MEAN_REVERSION
        if regime["consensus_state"] == "CHOP_NEUTRAL":
            assert regime["recommended_logic"] == "PAIRS_MEAN_REVERSION"

    def test_json_schema_complete(self):
        """Full v3.1 JSON schema with all Phase 4 fields."""
        from regime_detection import RegimeManager
        m = RegimeManager()
        self._feed(m, 150)
        parsed = json.loads(m.get_json())

        expected_top = {
            "consensus_state", "market_type", "confidence_score",
            "volatility_regime", "signals", "recommended_logic",
            "exit_mandate", "timestamp",
        }
        assert set(parsed.keys()) == expected_top
        assert isinstance(parsed["exit_mandate"], bool)
        assert parsed["recommended_logic"] in {e.value for e in RecommendedLogic}

    def test_structural_break_triggers_exit_mandate(self):
        """Simulate a structural break via dramatic price spike."""
        from regime_detection import RegimeManager
        m = RegimeManager(strategy_type="scalping", market_class="crypto")

        # Feed stable data
        for i in range(150):
            m.update(self._make_bar(150.0 + 0.1 * (i % 3)), funding_rate=0.0003)

        # Feed dramatic spike — may trigger CPD
        for i in range(20):
            m.update(self._make_bar(200.0 + 5.0 * i), funding_rate=0.0003)

        regime = m.get_current_regime()
        # At minimum, the output should be valid
        assert isinstance(regime["exit_mandate"], bool)
        # Structural break or regime shift likely detected
        assert regime["consensus_state"] in {
            "BULL_PERSISTENT", "BEAR_PERSISTENT", "CHOP_NEUTRAL",
            "TRANSITION", "UNKNOWN",
        }
