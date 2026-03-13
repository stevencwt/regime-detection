"""
regime_detection.schema — Output schema definitions for v3.1 spec.

Defines the enums, typed dictionaries, and serialization helpers that
guarantee every call to RegimeManager.get_json() produces the exact
standardized JSON structure from the specification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Enumerations matching v3.1 spec vocabulary
# ---------------------------------------------------------------------------

class ConsensusState(str, Enum):
    """Top-level regime classification."""
    BULL_PERSISTENT = "BULL_PERSISTENT"
    BEAR_PERSISTENT = "BEAR_PERSISTENT"
    CHOP_NEUTRAL = "CHOP_NEUTRAL"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


class MarketType(str, Enum):
    """Market category for the asset being analyzed."""
    CRYPTO_PERP = "CRYPTO_PERP"
    CRYPTO_SPOT = "CRYPTO_SPOT"
    US_STOCK = "US_STOCK"
    US_OPTION = "US_OPTION"
    SPREAD = "SPREAD"              # for pairs trading spread series


class VolatilityRegime(str, Enum):
    """Volatility classification."""
    LOW_STABLE = "LOW_STABLE"
    MODERATE = "MODERATE"
    EXPANDING = "EXPANDING"
    CONTRACTING = "CONTRACTING"
    UNKNOWN = "UNKNOWN"


class HMMLabel(str, Enum):
    """Hidden Markov Model state labels."""
    BULL = "BULL"
    BEAR = "BEAR"
    CHOP = "CHOP"
    UNKNOWN = "UNKNOWN"


class LiquidityStatus(str, Enum):
    """Order-book liquidity heuristic."""
    CONSOLIDATION = "CONSOLIDATION"
    LIQUIDITY_TRAP = "LIQUIDITY_TRAP"
    PASSED = "PASSED"
    UNKNOWN = "UNKNOWN"


class RecommendedLogic(str, Enum):
    """Downstream strategy recommendation — the key output."""
    SCALP_MEAN_REVERSION = "SCALP_MEAN_REVERSION"
    RANGE_TRADING = "RANGE_TRADING"
    SWING_TREND_FOLLOW = "SWING_TREND_FOLLOW"
    PAIRS_MEAN_REVERSION = "PAIRS_MEAN_REVERSION"
    OPTIONS_INCOME = "OPTIONS_INCOME"
    OPTIONS_SPECULATIVE = "OPTIONS_SPECULATIVE"
    NO_TRADE = "NO_TRADE"


class FundingBias(str, Enum):
    """Crypto perpetual funding rate bias."""
    EXTREME_POSITIVE = "EXTREME_POSITIVE"
    EXTREME_NEGATIVE = "EXTREME_NEGATIVE"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Dataclasses for structured output
# ---------------------------------------------------------------------------

@dataclass
class RangeHints:
    """Boundary hints for range-trading logic."""
    is_clean_range: bool = False
    range_lower: Optional[float] = None
    range_upper: Optional[float] = None
    current_deviation_pct: Optional[float] = None
    channel_type: Optional[str] = None


@dataclass
class CryptoContext:
    """Crypto-specific signals."""
    funding_bias: str = FundingBias.UNKNOWN.value
    funding_rate: Optional[float] = None
    order_book_imbalance: Optional[float] = None


@dataclass
class OptionsContext:
    """US-stock options context (None for crypto)."""
    vanna_signal: Optional[float] = None
    gamma_boundary: Optional[float] = None
    oi_skew: Optional[float] = None


@dataclass
class PairsContext:
    """Pairs-trading context."""
    spread_hurst: Optional[float] = None
    spread_half_life: Optional[float] = None
    cointegration_pvalue: Optional[float] = None


@dataclass
class Signals:
    """All individual signal components feeding the consensus."""
    hmm_label: str = HMMLabel.UNKNOWN.value
    hurst_dfa: Optional[float] = None
    structural_break: bool = False
    liquidity_status: str = LiquidityStatus.UNKNOWN.value
    crypto_context: Optional[CryptoContext] = None
    options_context: Optional[OptionsContext] = None
    pairs_context: Optional[PairsContext] = None
    range_hints: Optional[RangeHints] = None


@dataclass
class RegimeOutput:
    """Top-level output matching the v3.1 JSON schema exactly."""
    consensus_state: str = ConsensusState.UNKNOWN.value
    market_type: str = MarketType.CRYPTO_PERP.value
    confidence_score: float = 0.0
    volatility_regime: str = VolatilityRegime.UNKNOWN.value
    signals: Signals = field(default_factory=Signals)
    recommended_logic: str = RecommendedLogic.NO_TRADE.value
    exit_mandate: bool = False
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict (suitable for JSON serialization)."""
        d = asdict(self)
        # Remove None-valued optional contexts to keep output clean
        signals = d.get("signals", {})
        if signals.get("crypto_context") is None:
            del signals["crypto_context"]
        if signals.get("options_context") is None:
            del signals["options_context"]
        if signals.get("pairs_context") is None:
            del signals["pairs_context"]
        if signals.get("range_hints") is None:
            del signals["range_hints"]
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_default_output(market_type: MarketType = MarketType.CRYPTO_PERP) -> RegimeOutput:
    """Create a safe default RegimeOutput (UNKNOWN / NO_TRADE)."""
    return RegimeOutput(market_type=market_type.value)
