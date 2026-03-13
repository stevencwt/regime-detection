"""Tests for regime_detection.config — loading, merging, validation."""

import copy
import tempfile
from pathlib import Path

import pytest
import yaml

from regime_detection.config import (
    _deep_merge,
    load_config,
    load_default_config,
    validate_config,
)


# ---------------------------------------------------------------------------
# Default config loading
# ---------------------------------------------------------------------------

class TestLoadDefaultConfig:
    """Verify the bundled default_config.yaml loads correctly."""

    def test_loads_without_error(self):
        cfg = load_default_config()
        assert isinstance(cfg, dict)

    def test_has_required_sections(self):
        cfg = load_default_config()
        required = [
            "temporal_matrix", "hmm", "hurst", "cpd",
            "volatility", "liquidity", "funding",
            "range_detection", "exit_mandate",
        ]
        for section in required:
            assert section in cfg, f"Missing section: {section}"

    def test_version_is_3_1(self):
        cfg = load_default_config()
        assert cfg.get("version") == "3.1"

    def test_hmm_defaults(self):
        cfg = load_default_config()
        hmm = cfg["hmm"]
        assert hmm["n_states"] == 3
        assert hmm["covariance_type"] == "full"
        assert hmm["min_training_bars"] == 100

    def test_hurst_range_thresholds(self):
        cfg = load_default_config()
        h = cfg["hurst"]
        assert h["range_min_hurst"] == 0.48
        assert h["range_max_hurst"] == 0.58
        assert h["trending_threshold"] == 0.60

    def test_temporal_matrix_has_all_strategies(self):
        cfg = load_default_config()
        matrix = cfg["temporal_matrix"]
        expected = [
            "scalping", "range_trading", "swing",
            "options_income", "options_speculative", "pairs_trading",
        ]
        for key in expected:
            assert key in matrix, f"Missing strategy: {key}"

    def test_temporal_matrix_scalping_crypto(self):
        cfg = load_default_config()
        sc = cfg["temporal_matrix"]["scalping"]["crypto"]
        assert sc["regime_signal_tf"] == "5m"
        assert sc["execution_tf"] == "1m"
        assert sc["lookback_bars"] == 750

    def test_temporal_matrix_range_trading_crypto(self):
        cfg = load_default_config()
        rt = cfg["temporal_matrix"]["range_trading"]["crypto"]
        assert rt["regime_signal_tf"] == "15m"
        assert rt["lookback_bars"] == 1000
        assert rt["hmm_stability_bars"] == 3


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        over = {"b": 99}
        result = _deep_merge(base, over)
        assert result == {"a": 1, "b": 99}

    def test_nested_override(self):
        base = {"outer": {"a": 1, "b": 2}}
        over = {"outer": {"b": 99}}
        result = _deep_merge(base, over)
        assert result == {"outer": {"a": 1, "b": 99}}

    def test_adds_new_keys(self):
        base = {"a": 1}
        over = {"b": 2, "c": {"nested": True}}
        result = _deep_merge(base, over)
        assert result == {"a": 1, "b": 2, "c": {"nested": True}}

    def test_does_not_mutate_originals(self):
        base = {"a": {"x": 1}}
        over = {"a": {"y": 2}}
        base_copy = copy.deepcopy(base)
        _deep_merge(base, over)
        assert base == base_copy


# ---------------------------------------------------------------------------
# User override loading
# ---------------------------------------------------------------------------

class TestLoadConfigWithOverride:
    def test_override_single_value(self, tmp_path):
        override_file = tmp_path / "override.yaml"
        override_file.write_text(yaml.dump({"version": "custom_3.1.1"}))

        cfg = load_config(str(override_file))
        assert cfg["version"] == "custom_3.1.1"
        # Other defaults should still be present
        assert "hmm" in cfg

    def test_override_nested_value(self, tmp_path):
        override_file = tmp_path / "override.yaml"
        override_file.write_text(yaml.dump({
            "hmm": {"n_states": 5, "n_iter": 100}
        }))

        cfg = load_config(str(override_file))
        assert cfg["hmm"]["n_states"] == 5
        assert cfg["hmm"]["n_iter"] == 100
        # Untouched defaults survive
        assert cfg["hmm"]["covariance_type"] == "full"

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_none_path_returns_defaults(self):
        cfg = load_config(None)
        assert cfg["version"] == "3.1"

    def test_empty_override_file(self, tmp_path):
        override_file = tmp_path / "empty.yaml"
        override_file.write_text("")
        cfg = load_config(str(override_file))
        # Should just return defaults unchanged
        assert cfg["version"] == "3.1"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_default_config_passes(self):
        cfg = load_default_config()
        issues = validate_config(cfg)
        assert issues == [], f"Default config has issues: {issues}"

    def test_missing_section_detected(self):
        cfg = load_default_config()
        del cfg["hmm"]
        issues = validate_config(cfg)
        assert any("hmm" in i for i in issues)

    def test_invalid_hmm_n_states(self):
        cfg = load_default_config()
        cfg["hmm"]["n_states"] = 1
        issues = validate_config(cfg)
        assert any("n_states" in i for i in issues)

    def test_invalid_hurst_range(self):
        cfg = load_default_config()
        cfg["hurst"]["range_min_hurst"] = 0.70  # > range_max_hurst
        issues = validate_config(cfg)
        assert any("range" in i.lower() for i in issues)

    def test_invalid_cpd_penalty(self):
        cfg = load_default_config()
        cfg["cpd"]["penalty"] = -1
        issues = validate_config(cfg)
        assert any("penalty" in i for i in issues)

    def test_invalid_volatility_band(self):
        cfg = load_default_config()
        cfg["volatility"]["stable_band"] = [2.0, 0.5]  # reversed
        issues = validate_config(cfg)
        assert any("stable_band" in i for i in issues)

    def test_invalid_grace_bars(self):
        cfg = load_default_config()
        cfg["exit_mandate"]["grace_bars"] = -3
        issues = validate_config(cfg)
        assert any("grace_bars" in i for i in issues)
