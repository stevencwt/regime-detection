"""Tests for drift detection — compute_drift signal function.

Tests verify that directional drift is correctly classified:
  UP   — choppy uptrend (price above rising SMA)
  DOWN — choppy downtrend (price below falling SMA)
  NONE — true range-bound (price oscillates around flat SMA)
"""

from __future__ import annotations

import numpy as np
import pytest

from regime_detection.signals import compute_drift


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_DRIFT_CFG = {
    "sma_period": 50,
    "slope_window": 10,
    "strong_above_pct": 0.80,
    "moderate_above_pct": 0.65,
    "min_slope_pct": 0.15,
}


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _choppy_uptrend(n: int = 200, drift: float = 0.001, noise: float = 0.005) -> np.ndarray:
    """Price series with upward drift + mean-reverting noise.

    Hurst will be < 0.55 (choppy) but price consistently above rising SMA.
    """
    rng = np.random.RandomState(42)
    base = 100.0 * np.exp(np.cumsum(np.full(n, drift)))
    chop = noise * rng.randn(n)
    # Add mean-reverting overlay
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.7 * ou[i - 1] + chop[i]
    return base + base * ou


def _choppy_downtrend(n: int = 200, drift: float = -0.001, noise: float = 0.005) -> np.ndarray:
    """Price series with downward drift + mean-reverting noise."""
    rng = np.random.RandomState(42)
    base = 100.0 * np.exp(np.cumsum(np.full(n, drift)))
    chop = noise * rng.randn(n)
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.7 * ou[i - 1] + chop[i]
    return base + base * ou


def _flat_range(n: int = 200, center: float = 100.0, width: float = 2.0) -> np.ndarray:
    """Price series oscillating around a flat center — true range."""
    rng = np.random.RandomState(42)
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.9 * ou[i - 1] + 0.3 * rng.randn()
    return center + width * ou


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeDrift:
    """Tests for compute_drift()."""

    def test_uptrend_detected(self):
        """Choppy uptrend should return UP."""
        closes = _choppy_uptrend(200, drift=0.002)
        result = compute_drift(closes, DEFAULT_DRIFT_CFG)
        assert result == "UP", f"Expected UP, got {result}"

    def test_downtrend_detected(self):
        """Choppy downtrend should return DOWN."""
        closes = _choppy_downtrend(200, drift=-0.002)
        result = compute_drift(closes, DEFAULT_DRIFT_CFG)
        assert result == "DOWN", f"Expected DOWN, got {result}"

    def test_flat_range_is_none(self):
        """Flat oscillating range should return NONE."""
        closes = _flat_range(200)
        result = compute_drift(closes, DEFAULT_DRIFT_CFG)
        assert result == "NONE", f"Expected NONE, got {result}"

    def test_insufficient_data_returns_none(self):
        """Too few bars should return NONE."""
        closes = np.array([100.0] * 30)
        result = compute_drift(closes, DEFAULT_DRIFT_CFG)
        assert result == "NONE"

    def test_empty_array_returns_none(self):
        """Empty input should return NONE."""
        result = compute_drift(np.array([]), DEFAULT_DRIFT_CFG)
        assert result == "NONE"

    def test_strong_uptrend_above_80pct(self):
        """When 80%+ bars are above SMA, should return UP regardless of slope."""
        # Steady climb — nearly all bars above SMA
        closes = 100.0 + np.arange(200) * 0.5
        result = compute_drift(closes, DEFAULT_DRIFT_CFG)
        assert result == "UP"

    def test_strong_downtrend_below_80pct(self):
        """When 80%+ bars are below SMA, should return DOWN."""
        closes = 200.0 - np.arange(200) * 0.5
        result = compute_drift(closes, DEFAULT_DRIFT_CFG)
        assert result == "DOWN"

    def test_moderate_uptrend_needs_slope_confirm(self):
        """65-80% above SMA without slope should return NONE."""
        # Create series where price is above SMA ~70% but SMA is nearly flat
        rng = np.random.RandomState(123)
        closes = 100.0 + 0.5 * np.sin(np.linspace(0, 8 * np.pi, 200)) + 0.3
        result = compute_drift(closes, DEFAULT_DRIFT_CFG)
        # With flat SMA and slight offset, this should be NONE (no slope confirm)
        assert result == "NONE", f"Expected NONE, got {result}"

    def test_custom_thresholds(self):
        """Custom config thresholds should be respected."""
        # Use a mild drift that triggers default config but not strict config
        closes = _choppy_uptrend(200, drift=0.0003, noise=0.008)
        # Very strict threshold — should NOT trigger
        strict_cfg = {
            "sma_period": 50,
            "slope_window": 10,
            "strong_above_pct": 0.95,
            "moderate_above_pct": 0.90,
            "min_slope_pct": 1.0,
        }
        result = compute_drift(closes, strict_cfg)
        assert result == "NONE", f"Strict config should return NONE, got {result}"

    def test_default_config_values(self):
        """Empty config should use sensible defaults."""
        closes = _choppy_uptrend(200, drift=0.002)
        result = compute_drift(closes, {})
        # Should still detect UP with defaults
        assert result == "UP", f"Expected UP with defaults, got {result}"

    def test_swing_higher_lows_detects_early_uptrend(self):
        """Higher lows pattern should detect UP before SMA confirms."""
        # Create a series that starts flat then drifts up gradually.
        # SMA(50) will be slow to catch up, but swing lows will be rising.
        rng = np.random.RandomState(77)
        n = 100
        # First 60 bars: flat around 100
        flat = 100.0 + 0.5 * rng.randn(60)
        # Last 40 bars: staircase up with pullbacks (higher lows)
        staircase = np.zeros(40)
        level = flat[-1]
        for i in range(40):
            if i % 10 == 0 and i > 0:
                level += 1.5  # step up every 10 bars
            staircase[i] = level + 0.8 * np.sin(i * 0.5)  # oscillation
        closes = np.concatenate([flat, staircase])
        result = compute_drift(closes, DEFAULT_DRIFT_CFG)
        assert result == "UP", f"Swing higher-lows should detect UP, got {result}"

    def test_swing_lower_highs_detects_early_downtrend(self):
        """Lower highs pattern should detect DOWN before SMA confirms."""
        rng = np.random.RandomState(77)
        n = 100
        flat = 100.0 + 0.5 * rng.randn(60)
        staircase = np.zeros(40)
        level = flat[-1]
        for i in range(40):
            if i % 10 == 0 and i > 0:
                level -= 1.5
            staircase[i] = level + 0.8 * np.sin(i * 0.5)
        closes = np.concatenate([flat, staircase])
        result = compute_drift(closes, DEFAULT_DRIFT_CFG)
        assert result == "DOWN", f"Swing lower-highs should detect DOWN, got {result}"

    def test_no_swings_in_flat_range(self):
        """Flat range should NOT trigger swing detection."""
        closes = _flat_range(200)
        result = compute_drift(closes, DEFAULT_DRIFT_CFG)
        assert result == "NONE", f"Flat range should be NONE, got {result}"
