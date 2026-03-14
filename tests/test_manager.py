"""Tests for regime_detection.manager — RegimeManager skeleton behavior."""

import json
import time

import pytest
import yaml

from regime_detection import RegimeManager
from regime_detection.schema import (
    ConsensusState,
    MarketType,
    RecommendedLogic,
    VolatilityRegime,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_bar(close: float = 150.0, volume: float = 1000.0, ts: float = None) -> dict:
    """Create a valid OHLCV bar dict."""
    return {
        "timestamp": ts or time.time(),
        "o": close - 0.5,
        "h": close + 1.0,
        "l": close - 1.0,
        "c": close,
        "v": volume,
    }


def _feed_bars(manager: RegimeManager, n: int, base_close: float = 150.0):
    """Feed N synthetic bars into the manager."""
    for i in range(n):
        bar = _make_bar(close=base_close + (i % 5) * 0.1, ts=time.time() + i)
        manager.update(bar)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_init(self):
        m = RegimeManager()
        assert m.bar_count == 0
        assert m.config is not None
        assert m.config.get("version") == "3.1"

    def test_init_with_market_type(self):
        m = RegimeManager(market_type="US_STOCK", strategy_type="swing", market_class="us_stocks")
        regime = m.get_current_regime()
        assert regime["market_type"] == "US_STOCK"

    def test_init_with_strategy_type(self):
        m = RegimeManager(strategy_type="pairs_trading", market_class="crypto")
        assert m.temporal_settings["lookback_bars"] == 1000

    def test_init_with_custom_config(self, tmp_path):
        override = tmp_path / "custom.yaml"
        override.write_text(yaml.dump({
            "hmm": {"n_states": 5},
            "hurst": {"trending_threshold": 0.65},
        }))
        m = RegimeManager(config_path=str(override))
        assert m.config["hmm"]["n_states"] == 5
        assert m.config["hurst"]["trending_threshold"] == 0.65
        # Defaults still present
        assert m.config["hurst"]["range_min_hurst"] == 0.48

    def test_temporal_matrix_resolution(self):
        m = RegimeManager(strategy_type="scalping", market_class="crypto")
        t = m.temporal_settings
        assert t["regime_signal_tf"] == "5m"
        assert t["lookback_bars"] == 750

    def test_temporal_matrix_fallback_for_unknown(self):
        """Unknown strategy/market combo should get a safe fallback."""
        m = RegimeManager(strategy_type="nonexistent_strategy", market_class="mars")
        t = m.temporal_settings
        assert "lookback_bars" in t  # fallback was applied


# ---------------------------------------------------------------------------
# Bar ingestion and buffering
# ---------------------------------------------------------------------------

class TestBarIngestion:
    def test_valid_bar_increments_count(self):
        m = RegimeManager()
        m.update(_make_bar())
        assert m.bar_count == 1

    def test_multiple_bars(self):
        m = RegimeManager()
        _feed_bars(m, 10)
        assert m.bar_count == 10

    def test_invalid_bar_rejected(self):
        m = RegimeManager()
        m.update({"bad": "data"})
        assert m.bar_count == 0

    def test_missing_close_rejected(self):
        m = RegimeManager()
        m.update({"o": 1, "h": 2, "l": 0.5, "v": 100})  # no "c"
        assert m.bar_count == 0

    def test_none_bar_rejected(self):
        m = RegimeManager()
        m.update(None)
        assert m.bar_count == 0

    def test_buffer_respects_maxlen(self):
        """Buffer should cap at lookback_bars from temporal matrix."""
        m = RegimeManager(strategy_type="scalping", market_class="crypto")
        max_bars = m.temporal_settings["lookback_bars"]  # 750
        _feed_bars(m, max_bars + 50)
        assert m.bar_count == max_bars  # capped by deque maxlen


# ---------------------------------------------------------------------------
# Output during warmup
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_output_is_unknown_during_warmup(self):
        m = RegimeManager()
        _feed_bars(m, 10)  # well below min_training_bars=100
        regime = m.get_current_regime()
        assert regime["consensus_state"] == "UNKNOWN"
        assert regime["recommended_logic"] == "NO_TRADE"

    def test_json_valid_during_warmup(self):
        m = RegimeManager()
        _feed_bars(m, 5)
        j = m.get_json()
        parsed = json.loads(j)
        assert "consensus_state" in parsed
        assert "timestamp" in parsed


# ---------------------------------------------------------------------------
# Output after warmup (Phase 3: consensus is live, recommended_logic still stub)
# ---------------------------------------------------------------------------

class TestOutputAfterWarmup:
    def test_consensus_computed_after_warmup(self):
        """Phase 3: real signals + consensus produce a valid state after warmup."""
        m = RegimeManager()
        min_bars = m.config["hmm"]["min_training_bars"]
        _feed_bars(m, min_bars + 10)

        regime = m.get_current_regime()
        valid_states = {"BULL_PERSISTENT", "BEAR_PERSISTENT", "CHOP_NEUTRAL", "TRANSITION", "UNKNOWN"}
        assert regime["consensus_state"] in valid_states
        # recommended_logic still stub (Phase 4)
        assert regime["recommended_logic"] == "NO_TRADE"
        assert regime["exit_mandate"] is False

    def test_json_schema_keys_after_warmup(self):
        m = RegimeManager()
        _feed_bars(m, 120)
        parsed = json.loads(m.get_json())

        expected_top = {
            "consensus_state", "market_type", "confidence_score",
            "volatility_regime", "signals", "recommended_logic",
            "exit_mandate", "timestamp",
        }
        assert set(parsed.keys()) == expected_top

    def test_crypto_context_present_for_crypto_market(self):
        m = RegimeManager(market_type="CRYPTO_PERP")
        _feed_bars(m, 120)
        regime = m.get_current_regime()
        assert "crypto_context" in regime["signals"]

    def test_no_options_context_for_crypto(self):
        m = RegimeManager(market_type="CRYPTO_PERP")
        _feed_bars(m, 120)
        regime = m.get_current_regime()
        assert "options_context" not in regime["signals"]


# ---------------------------------------------------------------------------
# Update with optional data channels
# ---------------------------------------------------------------------------

class TestOptionalDataChannels:
    def test_update_with_funding_rate(self):
        """Funding rate accepted without error."""
        m = RegimeManager()
        m.update(_make_bar(), funding_rate=0.0003)
        assert m.bar_count == 1

    def test_update_with_order_book_imbalance(self):
        m = RegimeManager()
        m.update(_make_bar(), order_book_imbalance=-0.45)
        assert m.bar_count == 1

    def test_update_with_options_snapshot(self):
        m = RegimeManager(market_type="US_STOCK", strategy_type="swing", market_class="us_stocks")
        m.update(
            _make_bar(),
            options_snapshot={"vanna": 0.5, "gamma": -0.2},
        )
        assert m.bar_count == 1

    def test_update_with_spread_data(self):
        m = RegimeManager(market_type="SPREAD", strategy_type="pairs_trading", market_class="crypto")
        m.update(
            _make_bar(),
            spread_data={"spread_value": 0.003, "half_life": 22},
        )
        assert m.bar_count == 1

    def test_update_with_all_channels(self):
        m = RegimeManager()
        m.update(
            _make_bar(),
            funding_rate=0.0001,
            order_book_imbalance=0.15,
            options_snapshot=None,
            spread_data=None,
        )
        assert m.bar_count == 1


# ---------------------------------------------------------------------------
# Hot-reload
# ---------------------------------------------------------------------------

class TestHotReload:
    def test_reload_updates_config(self, tmp_path):
        override = tmp_path / "hot.yaml"
        override.write_text(yaml.dump({"hmm": {"n_states": 4}}))

        m = RegimeManager(config_path=str(override))
        assert m.config["hmm"]["n_states"] == 4

        # Modify the file
        override.write_text(yaml.dump({"hmm": {"n_states": 7}}))
        issues = m.reload_config()
        assert m.config["hmm"]["n_states"] == 7
        assert issues == []  # still valid

    def test_reload_preserves_bar_buffer(self, tmp_path):
        override = tmp_path / "hot.yaml"
        override.write_text(yaml.dump({}))

        m = RegimeManager(config_path=str(override))
        _feed_bars(m, 50)
        assert m.bar_count == 50

        m.reload_config()
        assert m.bar_count == 50  # buffer untouched
