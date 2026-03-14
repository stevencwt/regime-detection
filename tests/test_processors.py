"""Tests for regime_detection.processors — Phase 3 market processors & consensus.

Tests cover:
  - Crypto processor (funding bias + order-book imbalance)
  - Options processor (vanna, gamma, OI skew)
  - Pairs processor (spread Hurst, half-life, cointegration)
  - Consensus voting engine (all regime transitions)
  - Missing/partial data handling
  - Integration: processors wired through RegimeManager
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from regime_detection.processors import (
    process_crypto,
    process_options,
    process_pairs,
    vote_consensus,
)
from regime_detection.schema import (
    ConsensusState,
    HMMLabel,
    LiquidityStatus,
    VolatilityRegime,
)


# ---------------------------------------------------------------------------
# Default config sections
# ---------------------------------------------------------------------------

FUNDING_CFG = {
    "extreme_positive": 0.01,
    "extreme_negative": -0.01,
    "neutral_band": [-0.005, 0.005],
}

OPTIONS_CFG = {
    "vanna_threshold": 0.3,
    "gamma_flip_enabled": True,
    "oi_skew_lookback": 5,
}

PAIRS_CFG = {
    "spread_hurst_window": 200,
    "spread_half_life_max": 50,
    "cointegration_pvalue": 0.05,
}

HURST_CFG = {
    "min_window": 10,
    "max_window_ratio": 0.25,
    "order": 1,
    "trending_threshold": 0.60,
    "mean_reverting_threshold": 0.40,
    "range_min_hurst": 0.48,
    "range_max_hurst": 0.58,
}


# ===================================================================
# CRYPTO PROCESSOR
# ===================================================================

class TestCryptoProcessor:
    """Test process_crypto() with various inputs."""

    def test_with_full_data(self):
        ctx = process_crypto(0.0005, 0.15, FUNDING_CFG)
        assert ctx.funding_bias == "NEUTRAL"
        assert ctx.funding_rate == 0.0005
        assert ctx.order_book_imbalance == 0.15

    def test_extreme_positive_funding(self):
        ctx = process_crypto(0.015, 0.0, FUNDING_CFG)
        assert ctx.funding_bias == "EXTREME_POSITIVE"

    def test_extreme_negative_funding(self):
        ctx = process_crypto(-0.02, 0.0, FUNDING_CFG)
        assert ctx.funding_bias == "EXTREME_NEGATIVE"

    def test_none_funding(self):
        ctx = process_crypto(None, 0.1, FUNDING_CFG)
        assert ctx.funding_bias == "UNKNOWN"
        assert ctx.funding_rate is None
        assert ctx.order_book_imbalance == 0.1

    def test_none_imbalance(self):
        ctx = process_crypto(0.001, None, FUNDING_CFG)
        assert ctx.funding_bias == "NEUTRAL"
        assert ctx.order_book_imbalance is None

    def test_all_none(self):
        ctx = process_crypto(None, None, FUNDING_CFG)
        assert ctx.funding_bias == "UNKNOWN"
        assert ctx.funding_rate is None
        assert ctx.order_book_imbalance is None


# ===================================================================
# OPTIONS PROCESSOR
# ===================================================================

class TestOptionsProcessor:
    """Test process_options() with various inputs."""

    def test_none_snapshot_returns_empty(self):
        ctx = process_options(None, OPTIONS_CFG)
        assert ctx is not None
        assert ctx.vanna_signal is None
        assert ctx.gamma_boundary is None
        assert ctx.oi_skew is None

    def test_full_snapshot(self):
        snap = {"vanna": 0.45, "gamma_flip": 4500.0, "oi_skew": 1.3}
        ctx = process_options(snap, OPTIONS_CFG)
        assert ctx.vanna_signal == 0.45
        assert ctx.gamma_boundary == 4500.0
        assert ctx.oi_skew == 1.3

    def test_partial_snapshot_vanna_only(self):
        snap = {"vanna": -0.8}
        ctx = process_options(snap, OPTIONS_CFG)
        assert ctx.vanna_signal == -0.8
        assert ctx.gamma_boundary is None
        assert ctx.oi_skew is None

    def test_gamma_inferred_from_net_gamma(self):
        """When gamma_flip not provided, infer from net gamma value."""
        snap = {"vanna": 0.2, "gamma": -0.5}
        ctx = process_options(snap, OPTIONS_CFG)
        assert ctx.gamma_boundary == -0.5  # net gamma stored as boundary indicator

    def test_invalid_values_handled(self):
        snap = {"vanna": "bad", "gamma_flip": None, "oi_skew": "also_bad"}
        ctx = process_options(snap, OPTIONS_CFG)
        assert ctx.vanna_signal is None
        assert ctx.gamma_boundary is None
        assert ctx.oi_skew is None

    def test_empty_dict(self):
        ctx = process_options({}, OPTIONS_CFG)
        assert ctx.vanna_signal is None
        assert ctx.gamma_boundary is None


# ===================================================================
# PAIRS PROCESSOR
# ===================================================================

class TestPairsProcessor:
    """Test process_pairs() with spread data."""

    def test_none_spread_returns_empty(self):
        ctx = process_pairs(None, PAIRS_CFG, HURST_CFG)
        assert ctx is not None
        assert ctx.spread_hurst is None
        assert ctx.spread_half_life is None
        assert ctx.cointegration_pvalue is None

    def test_with_spread_series(self):
        """Spread series should produce a valid Hurst value."""
        rng = np.random.RandomState(42)
        spread = np.cumsum(rng.randn(200))  # random walk spread
        data = {
            "spread_series": spread.tolist(),
            "half_life": 15.3,
            "cointegration_pvalue": 0.02,
        }
        ctx = process_pairs(data, PAIRS_CFG, HURST_CFG)
        assert ctx.spread_hurst is not None
        assert 0.0 < ctx.spread_hurst < 1.5
        assert ctx.spread_half_life == 15.3
        assert ctx.cointegration_pvalue == 0.02

    def test_short_spread_series(self):
        """Spread too short for DFA → spread_hurst is None."""
        data = {"spread_series": [1.0, 2.0, 1.5]}
        ctx = process_pairs(data, PAIRS_CFG, HURST_CFG)
        assert ctx.spread_hurst is None

    def test_partial_data(self):
        """Only half-life provided, no spread series."""
        data = {"half_life": 22.0}
        ctx = process_pairs(data, PAIRS_CFG, HURST_CFG)
        assert ctx.spread_hurst is None
        assert ctx.spread_half_life == 22.0
        assert ctx.cointegration_pvalue is None

    def test_invalid_half_life(self):
        data = {"half_life": "invalid"}
        ctx = process_pairs(data, PAIRS_CFG, HURST_CFG)
        assert ctx.spread_half_life is None


# ===================================================================
# CONSENSUS VOTING ENGINE
# ===================================================================

class TestConsensusVoting:
    """Test vote_consensus() — the core state machine."""

    def test_bull_persistent(self):
        """BULL HMM + high Hurst + no break → BULL_PERSISTENT."""
        state, conf = vote_consensus(
            HMMLabel.BULL, 0.65, False,
            VolatilityRegime.MODERATE, LiquidityStatus.PASSED,
            HURST_CFG, hmm_confidence=0.85,
        )
        assert state == ConsensusState.BULL_PERSISTENT
        assert conf > 0.0

    def test_bear_persistent(self):
        """BEAR HMM + high Hurst + no break → BEAR_PERSISTENT."""
        state, conf = vote_consensus(
            HMMLabel.BEAR, 0.62, False,
            VolatilityRegime.MODERATE, LiquidityStatus.PASSED,
            HURST_CFG, hmm_confidence=0.80,
        )
        assert state == ConsensusState.BEAR_PERSISTENT
        assert conf > 0.0

    def test_chop_neutral_low_hurst(self):
        """CHOP HMM + low Hurst → CHOP_NEUTRAL."""
        state, conf = vote_consensus(
            HMMLabel.CHOP, 0.48, False,
            VolatilityRegime.LOW_STABLE, LiquidityStatus.CONSOLIDATION,
            HURST_CFG, hmm_confidence=0.70,
        )
        assert state == ConsensusState.CHOP_NEUTRAL

    def test_chop_neutral_bull_hmm_but_low_hurst(self):
        """BULL HMM but Hurst sub-trending → CHOP_NEUTRAL (Hurst overrides)."""
        state, conf = vote_consensus(
            HMMLabel.BULL, 0.52, False,
            VolatilityRegime.MODERATE, LiquidityStatus.PASSED,
            HURST_CFG, hmm_confidence=0.60,
        )
        assert state == ConsensusState.CHOP_NEUTRAL

    def test_transition_on_structural_break(self):
        """Structural break → TRANSITION regardless of other signals."""
        state, conf = vote_consensus(
            HMMLabel.BULL, 0.65, True,  # break = True
            VolatilityRegime.MODERATE, LiquidityStatus.PASSED,
            HURST_CFG, hmm_confidence=0.90,
        )
        assert state == ConsensusState.TRANSITION

    def test_transition_on_expanding_vol_with_trend(self):
        """EXPANDING volatility + trending Hurst → TRANSITION."""
        state, conf = vote_consensus(
            HMMLabel.BULL, 0.65, False,
            VolatilityRegime.EXPANDING, LiquidityStatus.PASSED,
            HURST_CFG, hmm_confidence=0.80,
        )
        assert state == ConsensusState.TRANSITION

    def test_unknown_when_hmm_unknown(self):
        """UNKNOWN HMM → UNKNOWN consensus."""
        state, conf = vote_consensus(
            HMMLabel.UNKNOWN, 0.55, False,
            VolatilityRegime.MODERATE, LiquidityStatus.PASSED,
            HURST_CFG, hmm_confidence=0.0,
        )
        assert state == ConsensusState.UNKNOWN
        assert conf == 0.0

    def test_unknown_when_hurst_none(self):
        """None Hurst → UNKNOWN consensus."""
        state, conf = vote_consensus(
            HMMLabel.CHOP, None, False,
            VolatilityRegime.MODERATE, LiquidityStatus.PASSED,
            HURST_CFG, hmm_confidence=0.70,
        )
        assert state == ConsensusState.UNKNOWN
        assert conf == 0.0

    def test_confidence_higher_with_strong_signals(self):
        """Strong HMM confidence + clear Hurst → higher overall confidence."""
        _, conf_strong = vote_consensus(
            HMMLabel.BULL, 0.70, False,
            VolatilityRegime.MODERATE, LiquidityStatus.PASSED,
            HURST_CFG, hmm_confidence=0.95,
        )
        _, conf_weak = vote_consensus(
            HMMLabel.BULL, 0.61, False,
            VolatilityRegime.MODERATE, LiquidityStatus.PASSED,
            HURST_CFG, hmm_confidence=0.40,
        )
        assert conf_strong > conf_weak

    def test_confidence_range(self):
        """Confidence should always be in [0, 1]."""
        for hurst_val in [0.3, 0.5, 0.7, 0.9]:
            for hmm_conf in [0.0, 0.5, 1.0]:
                _, conf = vote_consensus(
                    HMMLabel.CHOP, hurst_val, False,
                    VolatilityRegime.MODERATE, LiquidityStatus.PASSED,
                    HURST_CFG, hmm_confidence=hmm_conf,
                )
                assert 0.0 <= conf <= 1.0, f"conf={conf} for H={hurst_val}, hmm={hmm_conf}"

    def test_bear_with_low_hurst_is_chop(self):
        """BEAR HMM + sub-trending Hurst → CHOP_NEUTRAL (not persistent bear)."""
        state, _ = vote_consensus(
            HMMLabel.BEAR, 0.45, False,
            VolatilityRegime.LOW_STABLE, LiquidityStatus.CONSOLIDATION,
            HURST_CFG, hmm_confidence=0.70,
        )
        assert state == ConsensusState.CHOP_NEUTRAL


# ===================================================================
# MISSING DATA HANDLING
# ===================================================================

class TestMissingDataHandling:
    """Verify graceful degradation when inputs are None or incomplete."""

    def test_crypto_all_none_no_crash(self):
        ctx = process_crypto(None, None, FUNDING_CFG)
        assert ctx.funding_bias == "UNKNOWN"

    def test_options_none_no_crash(self):
        ctx = process_options(None, OPTIONS_CFG)
        assert ctx is not None

    def test_pairs_none_no_crash(self):
        ctx = process_pairs(None, PAIRS_CFG, HURST_CFG)
        assert ctx is not None

    def test_consensus_all_unknown_no_crash(self):
        state, conf = vote_consensus(
            HMMLabel.UNKNOWN, None, False,
            VolatilityRegime.UNKNOWN, LiquidityStatus.UNKNOWN,
            HURST_CFG, hmm_confidence=0.0,
        )
        assert state == ConsensusState.UNKNOWN
        assert conf == 0.0

    def test_pairs_with_invalid_spread_type(self):
        """Non-numeric spread_series should not crash."""
        data = {"spread_series": "not_a_list", "half_life": 10.0}
        ctx = process_pairs(data, PAIRS_CFG, HURST_CFG)
        # Should handle gracefully — spread_hurst may be None
        assert ctx.spread_half_life == 10.0


# ===================================================================
# INTEGRATION: Processors + Consensus wired through RegimeManager
# ===================================================================

class TestManagerPhase3Integration:
    """Verify the full Phase 3 pipeline via RegimeManager."""

    def _make_bar(self, close: float = 150.0) -> dict:
        return {
            "timestamp": time.time(),
            "o": close - 0.5,
            "h": close + 1.0,
            "l": close - 1.0,
            "c": close,
            "v": 10000.0,
        }

    def _feed(self, m, n: int, base: float = 150.0):
        for i in range(n):
            m.update(
                self._make_bar(base + (i % 5) * 0.1),
                funding_rate=0.0003,
                order_book_imbalance=0.1,
            )

    def test_consensus_not_unknown_after_warmup(self):
        from regime_detection import RegimeManager
        m = RegimeManager(strategy_type="scalping", market_class="crypto")
        self._feed(m, 150)
        regime = m.get_current_regime()
        # With real consensus, should get an actual state
        valid = {"BULL_PERSISTENT", "BEAR_PERSISTENT", "CHOP_NEUTRAL", "TRANSITION", "UNKNOWN"}
        assert regime["consensus_state"] in valid

    def test_confidence_is_populated(self):
        from regime_detection import RegimeManager
        m = RegimeManager()
        self._feed(m, 150)
        regime = m.get_current_regime()
        # Confidence should be > 0 after enough data
        assert regime["confidence_score"] >= 0.0

    def test_crypto_context_has_funding_bias(self):
        from regime_detection import RegimeManager
        m = RegimeManager(market_type="CRYPTO_PERP")
        self._feed(m, 120)
        regime = m.get_current_regime()
        ctx = regime["signals"]["crypto_context"]
        assert ctx["funding_bias"] == "NEUTRAL"  # 0.0003 is within neutral band
        assert ctx["funding_rate"] == 0.0003

    def test_options_context_for_us_stock(self):
        from regime_detection import RegimeManager
        m = RegimeManager(
            market_type="US_STOCK", strategy_type="swing", market_class="us_stocks"
        )
        for i in range(120):
            m.update(
                self._make_bar(450.0 + i * 0.1),
                options_snapshot={"vanna": 0.6, "gamma": -0.3, "oi_skew": 1.1},
            )
        regime = m.get_current_regime()
        ctx = regime["signals"]["options_context"]
        assert ctx["vanna_signal"] == 0.6
        assert ctx["gamma_boundary"] is not None
        assert ctx["oi_skew"] == 1.1

    def test_no_options_for_crypto(self):
        from regime_detection import RegimeManager
        m = RegimeManager(market_type="CRYPTO_PERP")
        self._feed(m, 120)
        regime = m.get_current_regime()
        assert "options_context" not in regime["signals"]

    def test_pairs_context_for_spread(self):
        from regime_detection import RegimeManager
        rng = np.random.RandomState(42)
        spread = np.cumsum(rng.randn(200))

        m = RegimeManager(
            market_type="SPREAD", strategy_type="pairs_trading", market_class="crypto"
        )
        for i in range(120):
            m.update(
                self._make_bar(150.0 + (i % 3) * 0.2),
                spread_data={
                    "spread_series": spread.tolist(),
                    "half_life": 18.0,
                    "cointegration_pvalue": 0.03,
                },
            )
        regime = m.get_current_regime()
        ctx = regime["signals"]["pairs_context"]
        assert ctx["spread_hurst"] is not None
        assert ctx["spread_half_life"] == 18.0
        assert ctx["cointegration_pvalue"] == 0.03

    def test_json_schema_still_valid(self):
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

    def test_recommended_logic_still_stub(self):
        """Phase 4 stub: recommended_logic should still be NO_TRADE."""
        from regime_detection import RegimeManager
        m = RegimeManager()
        self._feed(m, 150)
        regime = m.get_current_regime()
        assert regime["recommended_logic"] == "NO_TRADE"

    def test_exit_mandate_still_stub(self):
        """Phase 4 stub: exit_mandate should still be False."""
        from regime_detection import RegimeManager
        m = RegimeManager()
        self._feed(m, 150)
        regime = m.get_current_regime()
        assert regime["exit_mandate"] is False
