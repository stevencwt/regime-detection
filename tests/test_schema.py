"""Tests for regime_detection.schema — output types and JSON serialization."""

import json

from regime_detection.schema import (
    ConsensusState,
    CryptoContext,
    FundingBias,
    HMMLabel,
    LiquidityStatus,
    MarketType,
    OptionsContext,
    PairsContext,
    RangeHints,
    RecommendedLogic,
    RegimeOutput,
    Signals,
    VolatilityRegime,
    make_default_output,
)


class TestEnums:
    """Verify enum values match v3.1 spec strings exactly."""

    def test_consensus_states(self):
        assert ConsensusState.BULL_PERSISTENT.value == "BULL_PERSISTENT"
        assert ConsensusState.CHOP_NEUTRAL.value == "CHOP_NEUTRAL"

    def test_recommended_logic_includes_range_and_scalp(self):
        """v3.1 key requirement: RANGE_TRADING is distinct from SCALP_MEAN_REVERSION."""
        values = [e.value for e in RecommendedLogic]
        assert "SCALP_MEAN_REVERSION" in values
        assert "RANGE_TRADING" in values
        assert "SWING_TREND_FOLLOW" in values
        assert "PAIRS_MEAN_REVERSION" in values
        assert "NO_TRADE" in values

    def test_volatility_regimes(self):
        assert VolatilityRegime.LOW_STABLE.value == "LOW_STABLE"
        assert VolatilityRegime.EXPANDING.value == "EXPANDING"

    def test_funding_bias(self):
        assert FundingBias.EXTREME_POSITIVE.value == "EXTREME_POSITIVE"


class TestRegimeOutput:
    """Verify the top-level output dataclass."""

    def test_default_output_has_timestamp(self):
        out = make_default_output()
        assert out.timestamp != ""
        assert "T" in out.timestamp  # ISO format

    def test_default_is_unknown_no_trade(self):
        out = make_default_output()
        assert out.consensus_state == ConsensusState.UNKNOWN.value
        assert out.recommended_logic == RecommendedLogic.NO_TRADE.value
        assert out.exit_mandate is False
        assert out.confidence_score == 0.0

    def test_to_dict_is_plain_dict(self):
        out = make_default_output()
        d = out.to_dict()
        assert isinstance(d, dict)
        assert "consensus_state" in d
        assert "signals" in d
        assert "recommended_logic" in d

    def test_to_json_is_valid_json(self):
        out = make_default_output()
        s = out.to_json()
        parsed = json.loads(s)
        assert parsed["consensus_state"] == "UNKNOWN"
        assert parsed["recommended_logic"] == "NO_TRADE"

    def test_optional_contexts_stripped_when_none(self):
        out = make_default_output()
        d = out.to_dict()
        signals = d["signals"]
        # Default output has no options/pairs context
        assert "options_context" not in signals
        assert "pairs_context" not in signals

    def test_crypto_context_present_when_set(self):
        out = RegimeOutput(
            signals=Signals(
                crypto_context=CryptoContext(
                    funding_bias=FundingBias.EXTREME_POSITIVE.value,
                    funding_rate=0.0012,
                )
            )
        )
        d = out.to_dict()
        ctx = d["signals"]["crypto_context"]
        assert ctx["funding_bias"] == "EXTREME_POSITIVE"
        assert ctx["funding_rate"] == 0.0012


class TestRangeHints:
    """Verify range hint structure (v3.1 addition)."""

    def test_range_hints_in_json(self):
        out = RegimeOutput(
            consensus_state=ConsensusState.CHOP_NEUTRAL.value,
            recommended_logic=RecommendedLogic.RANGE_TRADING.value,
            signals=Signals(
                range_hints=RangeHints(
                    is_clean_range=True,
                    range_lower=145.20,
                    range_upper=152.80,
                    current_deviation_pct=-1.4,
                    channel_type="Donchian_30",
                )
            ),
        )
        d = out.to_dict()
        rh = d["signals"]["range_hints"]
        assert rh["is_clean_range"] is True
        assert rh["range_lower"] == 145.20
        assert rh["range_upper"] == 152.80
        assert rh["channel_type"] == "Donchian_30"


class TestJsonSchemaMatchesSpec:
    """Verify the output matches the exact v3.1 example JSON keys."""

    def test_all_top_level_keys_present(self):
        out = RegimeOutput(
            consensus_state=ConsensusState.CHOP_NEUTRAL.value,
            market_type=MarketType.CRYPTO_PERP.value,
            confidence_score=0.85,
            volatility_regime=VolatilityRegime.LOW_STABLE.value,
            signals=Signals(
                hmm_label=HMMLabel.CHOP.value,
                hurst_dfa=0.52,
                structural_break=False,
                liquidity_status=LiquidityStatus.CONSOLIDATION.value,
                crypto_context=CryptoContext(),
                range_hints=RangeHints(is_clean_range=True),
            ),
            recommended_logic=RecommendedLogic.RANGE_TRADING.value,
            exit_mandate=False,
        )
        parsed = json.loads(out.to_json())

        expected_keys = {
            "consensus_state", "market_type", "confidence_score",
            "volatility_regime", "signals", "recommended_logic",
            "exit_mandate", "timestamp",
        }
        assert set(parsed.keys()) == expected_keys

    def test_signals_keys(self):
        out = RegimeOutput(
            signals=Signals(
                hmm_label=HMMLabel.CHOP.value,
                hurst_dfa=0.52,
                structural_break=False,
                liquidity_status=LiquidityStatus.CONSOLIDATION.value,
                crypto_context=CryptoContext(),
                range_hints=RangeHints(),
            ),
        )
        parsed = json.loads(out.to_json())
        sig_keys = set(parsed["signals"].keys())

        # Must have these per spec
        assert "hmm_label" in sig_keys
        assert "hurst_dfa" in sig_keys
        assert "structural_break" in sig_keys
        assert "liquidity_status" in sig_keys
        assert "crypto_context" in sig_keys
        assert "range_hints" in sig_keys
