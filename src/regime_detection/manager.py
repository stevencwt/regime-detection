"""
regime_detection.manager — RegimeManager public API.

This is the single entry-point that downstream bots interact with:

    manager = RegimeManager(config_path="my_config.yaml")
    manager.update(bar, funding_rate=0.0003, order_book_imbalance=-0.12)
    output = manager.get_json()
    if manager.get_current_regime()["exit_mandate"]:
        # close all positions ...

Phase 1: Config, schema, bar buffer, .update()/.get_json() skeleton.
Phase 2: Core signals wired — HMM, DFA Hurst, CPD, volatility, liquidity, funding.
Phase 3: Market processors (crypto, options, pairs) + consensus voting.
Phase 4: Recommended logic (range vs scalp vs swing), range hints, exit mandate.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from regime_detection.config import load_config, validate_config
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
from regime_detection.signals import (
    classify_funding_bias,
    classify_liquidity,
    classify_volatility,
    compute_cpd,
    compute_hmm_labels,
    compute_hurst_dfa,
)
from regime_detection.processors import (
    process_crypto,
    process_options,
    process_pairs,
    vote_consensus,
)
from regime_detection.recommendation import (
    compute_range_hints,
    compute_range_persistence,
    determine_recommended_logic,
    evaluate_exit_mandate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bar type definition (what .update() expects)
# ---------------------------------------------------------------------------
# A "bar" is a plain dict with at minimum:
#   {"timestamp": <epoch_sec or ISO str>, "o": float, "h": float, "l": float,
#    "c": float, "v": float}
# Additional keys are silently ignored.

_REQUIRED_BAR_KEYS = {"o", "h", "l", "c", "v"}


def _validate_bar(bar: Dict[str, Any]) -> bool:
    """Return True if bar dict has the required OHLCV keys with valid numbers."""
    if not isinstance(bar, dict):
        return False
    for key in _REQUIRED_BAR_KEYS:
        val = bar.get(key)
        if val is None:
            return False
        try:
            float(val)
        except (TypeError, ValueError):
            return False
    return True


# ---------------------------------------------------------------------------
# RegimeManager
# ---------------------------------------------------------------------------

class RegimeManager:
    """Multi-Modal Market Regime Analysis — public controller.

    Parameters
    ----------
    config_path : str | Path | None
        Path to a user YAML override file.  If None, built-in defaults are used.
    market_type : str
        One of the MarketType enum values (e.g. "CRYPTO_PERP", "US_STOCK").
    strategy_type : str
        Key into temporal_matrix (e.g. "scalping", "range_trading", "swing",
        "pairs_trading", "options_income", "options_speculative").
    market_class : str
        Sub-key under strategy type — "crypto" or "us_stocks".
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        market_type: str = "CRYPTO_PERP",
        strategy_type: str = "scalping",
        market_class: str = "crypto",
    ):
        # --- Config ---
        self._cfg = load_config(config_path)
        issues = validate_config(self._cfg)
        if issues:
            logger.warning(
                "Config loaded with %d validation issue(s) — see log for details",
                len(issues),
            )
        self._config_path = config_path

        # --- Market & strategy context ---
        self._market_type = MarketType(market_type)
        self._strategy_type = strategy_type
        self._market_class = market_class

        # Resolve temporal settings for this strategy/market combo
        self._temporal = self._resolve_temporal_settings()

        # --- Bar buffer (FIFO ring) ---
        max_lookback = self._temporal.get("lookback_bars", 1000)
        self._bar_buffer: deque[Dict[str, Any]] = deque(maxlen=max_lookback)
        self._close_buffer: deque[float] = deque(maxlen=max_lookback)

        # --- Latest regime output ---
        self._current_output: RegimeOutput = make_default_output(self._market_type)
        self._previous_output: Optional[RegimeOutput] = None

        # --- Regime shift tracking (for exit_mandate grace period) ---
        self._shift_counter: int = 0
        self._grace_bars: int = self._cfg.get("exit_mandate", {}).get("grace_bars", 2)

        # --- Update counter ---
        self._tick_count: int = 0

        # --- HMM state tracking (Phase 2) ---
        self._hmm_last_label: str = "UNKNOWN"
        self._hmm_confidence: float = 0.0

        # --- Exit mandate state tracking (Phase 4) ---
        self._previous_consensus: Optional[ConsensusState] = None
        self._range_persistence: int = 0

        logger.info(
            "RegimeManager initialized | market=%s strategy=%s/%s lookback=%d",
            self._market_type.value,
            self._strategy_type,
            self._market_class,
            max_lookback,
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _resolve_temporal_settings(self) -> Dict[str, Any]:
        """Look up the temporal matrix entry for the configured strategy/market."""
        matrix = self._cfg.get("temporal_matrix", {})
        strategy_block = matrix.get(self._strategy_type, {})
        settings = strategy_block.get(self._market_class, {})

        if not settings:
            logger.warning(
                "No temporal_matrix entry for %s/%s — using fallback defaults",
                self._strategy_type,
                self._market_class,
            )
            settings = {
                "regime_signal_tf": "5m",
                "execution_tf": "1m",
                "lookback_bars": 750,
                "hmm_stability_bars": 2,
            }
        return settings

    @property
    def config(self) -> Dict[str, Any]:
        """Read-only access to the merged config dict."""
        return self._cfg

    @property
    def temporal_settings(self) -> Dict[str, Any]:
        """Resolved temporal matrix entry for this instance."""
        return self._temporal

    @property
    def bar_count(self) -> int:
        """Number of bars currently in the buffer."""
        return len(self._bar_buffer)

    def reload_config(self, config_path: Optional[str | Path] = None) -> List[str]:
        """Hot-reload configuration from disk.

        Returns list of validation issues (empty = OK).
        """
        path = config_path or self._config_path
        self._cfg = load_config(path)
        issues = validate_config(self._cfg)
        self._temporal = self._resolve_temporal_settings()
        self._grace_bars = self._cfg.get("exit_mandate", {}).get("grace_bars", 2)
        logger.info("Config reloaded (%d issues)", len(issues))
        return issues

    # ------------------------------------------------------------------
    # Public API: .update()
    # ------------------------------------------------------------------

    def update(
        self,
        bar: Dict[str, Any],
        *,
        funding_rate: Optional[float] = None,
        order_book_imbalance: Optional[float] = None,
        options_snapshot: Optional[Dict[str, Any]] = None,
        spread_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Ingest one new bar and recompute the regime.

        Parameters
        ----------
        bar : dict
            OHLCV bar: {"timestamp": ..., "o": float, "h": float,
            "l": float, "c": float, "v": float}.
        funding_rate : float | None
            Current perpetual funding rate (crypto only).
        order_book_imbalance : float | None
            Order-book imbalance metric in [-1, +1].
        options_snapshot : dict | None
            Options greeks/OI data (US stocks only).
        spread_data : dict | None
            Spread series data for pairs trading.
        """
        # --- Validate bar ---
        if not _validate_bar(bar):
            logger.warning("Invalid bar rejected: %s", bar)
            return

        # --- Append to buffer ---
        self._bar_buffer.append(bar)
        self._close_buffer.append(float(bar["c"]))
        self._tick_count += 1

        # --- Check minimum data ---
        min_bars = self._cfg.get("hmm", {}).get("min_training_bars", 100)
        if len(self._bar_buffer) < min_bars:
            logger.debug(
                "Warming up: %d/%d bars collected", len(self._bar_buffer), min_bars
            )
            # Keep default UNKNOWN output during warmup
            self._current_output.timestamp = (
                datetime.now(timezone.utc).isoformat(timespec="seconds")
            )
            return

        # --- Phase 2+ : Compute signals ---
        # Each of these will be implemented in subsequent phases.
        # For now they return safe defaults.
        hmm_label = self._compute_hmm()
        hurst_value = self._compute_hurst()
        structural_break = self._compute_cpd()
        vol_regime = self._compute_volatility_regime()
        liquidity = self._compute_liquidity(order_book_imbalance)

        # --- Market-specific processors (Phase 3) ---
        crypto_ctx = self._process_crypto(funding_rate, order_book_imbalance)
        options_ctx = self._process_options(options_snapshot)
        pairs_ctx = self._process_pairs(spread_data)

        # --- Consensus voting (Phase 3) ---
        consensus, confidence = self._vote_consensus(
            hmm_label, hurst_value, structural_break, vol_regime, liquidity
        )

        # --- Recommended logic (Phase 4: range vs scalp) ---
        self._range_persistence = self._compute_range_persistence()
        recommended = self._determine_recommended_logic(
            consensus, hmm_label, hurst_value, vol_regime, liquidity,
            structural_break,
        )

        # --- Range hints (Phase 4) ---
        range_hints = self._compute_range_hints(consensus, recommended)

        # --- Exit mandate (Phase 4) ---
        exit_mandate = self._evaluate_exit_mandate(
            consensus, hurst_value, structural_break, vol_regime,
        )

        # --- Build output ---
        self._previous_output = self._current_output
        self._current_output = RegimeOutput(
            consensus_state=consensus.value,
            market_type=self._market_type.value,
            confidence_score=round(confidence, 4),
            volatility_regime=vol_regime.value,
            signals=Signals(
                hmm_label=hmm_label.value,
                hurst_dfa=hurst_value,
                structural_break=structural_break,
                liquidity_status=liquidity.value,
                crypto_context=crypto_ctx,
                options_context=options_ctx,
                pairs_context=pairs_ctx,
                range_hints=range_hints,
            ),
            recommended_logic=recommended.value,
            exit_mandate=exit_mandate,
        )

        # --- Track consensus for exit mandate grace period ---
        self._previous_consensus = consensus

    # ------------------------------------------------------------------
    # Public API: .get_current_regime() / .get_json()
    # ------------------------------------------------------------------

    def get_current_regime(self) -> Dict[str, Any]:
        """Return the latest regime as a plain dict."""
        return self._current_output.to_dict()

    def get_json(self, indent: int = 2) -> str:
        """Return the latest regime as a JSON string (v3.1 schema)."""
        return self._current_output.to_json(indent=indent)

    # ------------------------------------------------------------------
    # Signal computation (Phase 2 — live implementations)
    # ------------------------------------------------------------------

    def _compute_hmm(self) -> HMMLabel:
        """Fit GaussianHMM to close prices, return stable label."""
        closes = np.array(self._close_buffer, dtype=float)
        hmm_cfg = self._cfg.get("hmm", {})
        stability_bars = self._temporal.get("hmm_stability_bars", 2)

        label_str, _states, confidence = compute_hmm_labels(
            closes, hmm_cfg, stability_bars
        )
        self._hmm_last_label = label_str
        self._hmm_confidence = confidence

        try:
            return HMMLabel(label_str)
        except ValueError:
            return HMMLabel.UNKNOWN

    def _compute_hurst(self) -> Optional[float]:
        """Compute DFA Hurst exponent from close buffer."""
        closes = np.array(self._close_buffer, dtype=float)
        hurst_cfg = self._cfg.get("hurst", {})

        if len(closes) < 30:
            return None

        value = compute_hurst_dfa(closes, hurst_cfg)
        return round(value, 4)

    def _compute_cpd(self) -> bool:
        """Detect structural breaks via BinSeg CPD."""
        closes = np.array(self._close_buffer, dtype=float)
        cpd_cfg = self._cfg.get("cpd", {})

        structural_break, _breakpoints = compute_cpd(closes, cpd_cfg)
        return structural_break

    def _compute_volatility_regime(self) -> VolatilityRegime:
        """Classify current volatility from close buffer."""
        closes = np.array(self._close_buffer, dtype=float)
        vol_cfg = self._cfg.get("volatility", {})

        label_str = classify_volatility(closes, vol_cfg)

        try:
            return VolatilityRegime(label_str)
        except ValueError:
            return VolatilityRegime.UNKNOWN

    def _compute_liquidity(
        self, order_book_imbalance: Optional[float]
    ) -> LiquidityStatus:
        """Classify liquidity from order-book imbalance."""
        liq_cfg = self._cfg.get("liquidity", {})
        label_str = classify_liquidity(order_book_imbalance, liq_cfg)

        try:
            return LiquidityStatus(label_str)
        except ValueError:
            return LiquidityStatus.UNKNOWN

    # ------------------------------------------------------------------
    # Market-specific processors (Phase 3 — live implementations)
    # ------------------------------------------------------------------

    def _process_crypto(
        self,
        funding_rate: Optional[float],
        order_book_imbalance: Optional[float],
    ) -> Optional[CryptoContext]:
        """Build crypto context from funding + order-book data."""
        if self._market_type not in (MarketType.CRYPTO_PERP, MarketType.CRYPTO_SPOT):
            return None

        funding_cfg = self._cfg.get("funding", {})
        return process_crypto(funding_rate, order_book_imbalance, funding_cfg)

    def _process_options(
        self, options_snapshot: Optional[Dict[str, Any]]
    ) -> Optional[OptionsContext]:
        """Build options context from greeks/OI snapshot."""
        if self._market_type not in (MarketType.US_STOCK, MarketType.US_OPTION):
            return None

        options_cfg = self._cfg.get("options", {})
        return process_options(options_snapshot, options_cfg)

    def _process_pairs(
        self, spread_data: Optional[Dict[str, Any]]
    ) -> Optional[PairsContext]:
        """Build pairs context from spread series data."""
        if self._market_type != MarketType.SPREAD:
            return None

        pairs_cfg = self._cfg.get("pairs", {})
        hurst_cfg = self._cfg.get("hurst", {})
        return process_pairs(spread_data, pairs_cfg, hurst_cfg)

    # ------------------------------------------------------------------
    # Consensus voting (Phase 3 — live implementation)
    # ------------------------------------------------------------------

    def _vote_consensus(
        self,
        hmm_label: HMMLabel,
        hurst_value: Optional[float],
        structural_break: bool,
        vol_regime: VolatilityRegime,
        liquidity: LiquidityStatus,
    ) -> tuple[ConsensusState, float]:
        """Vote on consensus regime state from all signals."""
        hurst_cfg = self._cfg.get("hurst", {})
        return vote_consensus(
            hmm_label,
            hurst_value,
            structural_break,
            vol_regime,
            liquidity,
            hurst_cfg,
            hmm_confidence=self._hmm_confidence,
        )

    # ------------------------------------------------------------------
    # Recommendation, range hints, exit mandate (Phase 4)
    # ------------------------------------------------------------------

    def _compute_range_persistence(self) -> int:
        """Count consecutive bars inside the channel."""
        closes = np.array(self._close_buffer, dtype=float)
        range_cfg = self._cfg.get("range_detection", {})
        return compute_range_persistence(closes, range_cfg)

    def _determine_recommended_logic(
        self,
        consensus: ConsensusState,
        hmm_label: HMMLabel,
        hurst_value: Optional[float],
        vol_regime: VolatilityRegime,
        liquidity: LiquidityStatus,
        structural_break: bool,
    ) -> RecommendedLogic:
        """Determine recommended execution logic per v3.1 activation rules."""
        hurst_cfg = self._cfg.get("hurst", {})
        range_cfg = self._cfg.get("range_detection", {})

        return determine_recommended_logic(
            consensus=consensus,
            hmm_label=hmm_label,
            hurst_value=hurst_value,
            vol_regime=vol_regime,
            liquidity=liquidity,
            structural_break=structural_break,
            strategy_type=self._strategy_type,
            hurst_cfg=hurst_cfg,
            range_cfg=range_cfg,
            range_persistence_bars=self._range_persistence,
        )

    def _compute_range_hints(
        self,
        consensus: ConsensusState,
        recommended: RecommendedLogic,
    ) -> Optional[RangeHints]:
        """Compute range boundary hints for range/scalp regimes."""
        closes = np.array(self._close_buffer, dtype=float)
        range_cfg = self._cfg.get("range_detection", {})

        return compute_range_hints(closes, consensus, recommended, range_cfg)

    def _evaluate_exit_mandate(
        self,
        new_consensus: ConsensusState,
        hurst_value: Optional[float],
        structural_break: bool,
        vol_regime: VolatilityRegime,
    ) -> bool:
        """Evaluate exit mandate with grace period and immediate triggers."""
        exit_cfg = self._cfg.get("exit_mandate", {})
        hurst_cfg = self._cfg.get("hurst", {})

        mandate, new_counter = evaluate_exit_mandate(
            current_consensus=new_consensus,
            previous_consensus=self._previous_consensus,
            shift_counter=self._shift_counter,
            grace_bars=self._grace_bars,
            exit_cfg=exit_cfg,
            hurst_value=hurst_value,
            structural_break=structural_break,
            vol_regime=vol_regime,
            hurst_cfg=hurst_cfg,
        )

        self._shift_counter = new_counter
        return mandate
