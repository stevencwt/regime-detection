"""
regime_detection.recommendation — Strategy recommendation & exit mandate (Phase 4).

Pure functions that:
  1. Map consensus + signals → recommended_logic (v3.1 activation rules)
  2. Compute range boundary hints (Donchian/Keltner channels)
  3. Evaluate exit mandate on regime shift with grace period

All functions are stateless and pure — state tracking (shift counter,
previous consensus) lives in the RegimeManager.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from regime_detection.schema import (
    ConsensusState,
    DriftDirection,
    HMMLabel,
    LiquidityStatus,
    RangeHints,
    RecommendedLogic,
    VolatilityRegime,
)

logger = logging.getLogger(__name__)


# ===================================================================
# 1. RECOMMENDED LOGIC — v3.1 ACTIVATION RULES
# ===================================================================

def determine_recommended_logic(
    consensus: ConsensusState,
    hmm_label: HMMLabel,
    hurst_value: Optional[float],
    vol_regime: VolatilityRegime,
    liquidity: LiquidityStatus,
    structural_break: bool,
    strategy_type: str,
    hurst_cfg: Dict,
    range_cfg: Dict,
    range_persistence_bars: int = 0,
    drift: DriftDirection = DriftDirection.NONE,
) -> RecommendedLogic:
    """Determine the recommended execution logic per v3.1 spec Section 5.

    Activation rules (evaluated in priority order):

    1. TRANSITION / UNKNOWN / structural break → NO_TRADE
    2. Strategy-specific overrides (pairs, options)
    3. BULL/BEAR_PERSISTENT → SWING_TREND_FOLLOW
    4. CHOP_NEUTRAL → SCALP vs RANGE vs NO_TRADE

    Parameters
    ----------
    consensus : ConsensusState
    hmm_label : HMMLabel
    hurst_value : float or None
    vol_regime : VolatilityRegime
    liquidity : LiquidityStatus
    structural_break : bool
    strategy_type : str — "scalping", "range_trading", "swing", "pairs_trading", etc.
    hurst_cfg : dict — hurst section (thresholds)
    range_cfg : dict — range_detection section
    range_persistence_bars : int — how many consecutive bars price stayed in channel

    Returns
    -------
    RecommendedLogic
    """
    # --- Safety gates ---
    if consensus == ConsensusState.UNKNOWN:
        return RecommendedLogic.NO_TRADE

    if consensus == ConsensusState.TRANSITION:
        return RecommendedLogic.NO_TRADE

    if structural_break:
        return RecommendedLogic.NO_TRADE

    if hurst_value is None:
        return RecommendedLogic.NO_TRADE

    # --- Strategy-specific overrides ---
    if strategy_type == "pairs_trading":
        # Pairs trading: needs CHOP (mean-reverting spread)
        if consensus == ConsensusState.CHOP_NEUTRAL:
            return RecommendedLogic.PAIRS_MEAN_REVERSION
        else:
            # Trending market → pairs spread may diverge, not safe
            return RecommendedLogic.NO_TRADE

    if strategy_type == "options_income":
        # Options income: stable/contracting vol preferred
        if vol_regime in (VolatilityRegime.LOW_STABLE, VolatilityRegime.MODERATE,
                          VolatilityRegime.CONTRACTING):
            return RecommendedLogic.OPTIONS_INCOME
        else:
            return RecommendedLogic.NO_TRADE

    if strategy_type == "options_speculative":
        # Options speculative: expanding vol or trending = opportunity
        if consensus in (ConsensusState.BULL_PERSISTENT, ConsensusState.BEAR_PERSISTENT):
            return RecommendedLogic.OPTIONS_SPECULATIVE
        if vol_regime == VolatilityRegime.EXPANDING:
            return RecommendedLogic.OPTIONS_SPECULATIVE
        return RecommendedLogic.NO_TRADE

    # --- Trending regimes → SWING ---
    if consensus in (ConsensusState.BULL_PERSISTENT, ConsensusState.BEAR_PERSISTENT):
        return RecommendedLogic.SWING_TREND_FOLLOW

    # --- CHOP_NEUTRAL → distinguish SCALP vs RANGE ---
    if consensus == ConsensusState.CHOP_NEUTRAL:
        return _classify_chop_sub_regime(
            hurst_value, vol_regime, liquidity,
            hurst_cfg, range_cfg, range_persistence_bars,
            drift,
        )

    # Fallback
    return RecommendedLogic.NO_TRADE


def _classify_chop_sub_regime(
    hurst_value: float,
    vol_regime: VolatilityRegime,
    liquidity: LiquidityStatus,
    hurst_cfg: Dict,
    range_cfg: Dict,
    range_persistence_bars: int,
    drift: DriftDirection = DriftDirection.NONE,
) -> RecommendedLogic:
    """Within CHOP_NEUTRAL, decide between SCALP and RANGE per v3.1 Section 5.

    RANGE_TRADING activation (all must be true):
      - Hurst in [range_min, range_max] (0.48–0.58 sweet spot)
      - Vol = LOW_STABLE or CONTRACTING
      - Liquidity = CONSOLIDATION, PASSED, or UNKNOWN (no OB data ≠ trap)
      - Range persistence >= min_bars_persistence
      - Drift = NONE (a range with directional drift is NOT a clean range)

    SCALP_MEAN_REVERSION activation:
      - Hurst < range_min (0.48) — noisy chop, not clean range
      - Liquidity not a LIQUIDITY_TRAP
      - No requirement on vol stability
      - Works with any drift direction (strategy layer handles direction filtering)

    Otherwise → NO_TRADE (chop but conditions not met for either)

    Note: UNKNOWN liquidity (no order-book data available) is treated as
    non-blocking. Only an explicit LIQUIDITY_TRAP blocks trading.
    """
    range_min_h = hurst_cfg.get("range_min_hurst", 0.48)
    range_max_h = hurst_cfg.get("range_max_hurst", 0.58)
    min_persistence = range_cfg.get("min_bars_persistence", 10)

    # Liquidity checks: UNKNOWN = no data, not a trap → non-blocking
    liq_not_trap = liquidity != LiquidityStatus.LIQUIDITY_TRAP

    # --- RANGE TRADING check (stricter conditions) ---
    hurst_in_range = range_min_h <= hurst_value <= range_max_h
    vol_stable = vol_regime in (VolatilityRegime.LOW_STABLE, VolatilityRegime.CONTRACTING)
    range_persists = range_persistence_bars >= min_persistence
    no_drift = drift == DriftDirection.NONE

    if hurst_in_range and vol_stable and liq_not_trap and range_persists and no_drift:
        return RecommendedLogic.RANGE_TRADING

    # --- SCALP check (looser conditions) ---
    hurst_below_range = hurst_value < range_min_h

    if hurst_below_range and liq_not_trap:
        return RecommendedLogic.SCALP_MEAN_REVERSION

    # --- Hurst in range sweet spot but persistence not met yet ---
    # Could be developing range — allow scalping as interim
    if hurst_in_range and liq_not_trap:
        return RecommendedLogic.SCALP_MEAN_REVERSION

    # --- LIQUIDITY_TRAP or other blocking condition ---
    return RecommendedLogic.NO_TRADE


# ===================================================================
# 2. RANGE HINTS — DONCHIAN / KELTNER CHANNEL BOUNDARIES
# ===================================================================

def compute_range_hints(
    closes: np.ndarray,
    consensus: ConsensusState,
    recommended: RecommendedLogic,
    range_cfg: Dict,
) -> Optional[RangeHints]:
    """Compute range boundary hints for range-trading guidance.

    Only computed when consensus is CHOP_NEUTRAL and recommended logic
    is RANGE_TRADING or SCALP_MEAN_REVERSION.

    Parameters
    ----------
    closes : 1-D array of close prices (oldest → newest)
    consensus : current ConsensusState
    recommended : current RecommendedLogic
    range_cfg : range_detection section from config

    Returns
    -------
    RangeHints or None (if not applicable)
    """
    # Only generate hints for chop regimes with active mean-reversion logic
    if consensus != ConsensusState.CHOP_NEUTRAL:
        return None

    if recommended not in (RecommendedLogic.RANGE_TRADING,
                           RecommendedLogic.SCALP_MEAN_REVERSION):
        return None

    if not range_cfg.get("enabled", True):
        return None

    channel_type = range_cfg.get("channel_type", "donchian")
    period = range_cfg.get("channel_period", 30)

    if len(closes) < period:
        return None

    window = closes[-period:]
    current_price = closes[-1]

    # --- Compute channel boundaries ---
    if channel_type == "donchian":
        range_upper = float(np.max(window))
        range_lower = float(np.min(window))
        channel_label = f"Donchian_{period}"
    elif channel_type == "keltner":
        # Keltner: EMA center ± ATR multiplier
        ema = float(np.mean(window))  # simplified as SMA
        atr = float(np.mean(np.abs(np.diff(window))))  # simplified ATR
        multiplier = 2.0
        range_upper = ema + multiplier * atr
        range_lower = ema - multiplier * atr
        channel_label = f"Keltner_{period}"
    else:
        range_upper = float(np.max(window))
        range_lower = float(np.min(window))
        channel_label = f"{channel_type}_{period}"

    # --- Current deviation from range midpoint ---
    range_mid = (range_upper + range_lower) / 2.0
    range_width = range_upper - range_lower

    if range_width > 0:
        deviation_pct = ((current_price - range_mid) / (range_width / 2.0)) * 100.0
    else:
        deviation_pct = 0.0

    # --- Is it a "clean" range? ---
    # Clean = recommended is RANGE_TRADING (all conditions met)
    is_clean = (recommended == RecommendedLogic.RANGE_TRADING)

    return RangeHints(
        is_clean_range=is_clean,
        range_lower=round(range_lower, 4),
        range_upper=round(range_upper, 4),
        current_deviation_pct=round(deviation_pct, 2),
        channel_type=channel_label,
    )


# ===================================================================
# 3. RANGE PERSISTENCE COUNTER
# ===================================================================

def compute_range_persistence(
    closes: np.ndarray,
    range_cfg: Dict,
) -> int:
    """Count how many recent consecutive bars stayed inside the channel.

    Scans backward from the most recent bar.  Returns 0 if the current
    bar is outside the channel.

    Parameters
    ----------
    closes : 1-D array of close prices
    range_cfg : range_detection section from config

    Returns
    -------
    int : number of consecutive bars inside the range (from most recent)
    """
    period = range_cfg.get("channel_period", 30)

    if len(closes) < period + 1:
        return 0

    # Compute channel from the lookback window BEFORE the recent bars
    # We use a sliding approach: channel defined by the first `period` bars,
    # then count how many subsequent bars stayed inside
    channel_base = closes[-(period + 20):-(20)] if len(closes) > period + 20 else closes[:period]

    if len(channel_base) < period:
        channel_base = closes[:period]

    upper = float(np.max(channel_base))
    lower = float(np.min(channel_base))

    # Add a small tolerance (0.5% of range width)
    width = upper - lower
    tolerance = width * 0.005 if width > 0 else 0.0

    # Scan backward from most recent bar
    count = 0
    for i in range(len(closes) - 1, -1, -1):
        price = closes[i]
        if (lower - tolerance) <= price <= (upper + tolerance):
            count += 1
        else:
            break

    return count


# ===================================================================
# 4. EXIT MANDATE — REGIME SHIFT DETECTION
# ===================================================================

def evaluate_exit_mandate(
    current_consensus: ConsensusState,
    previous_consensus: Optional[ConsensusState],
    shift_counter: int,
    grace_bars: int,
    exit_cfg: Dict,
    hurst_value: Optional[float],
    structural_break: bool,
    vol_regime: VolatilityRegime,
    hurst_cfg: Dict,
) -> Tuple[bool, int]:
    """Evaluate whether an exit mandate should be issued.

    Per v3.1 spec Section 5 "Global Mandatory Exit Rule":
      Any regime shift (CHOP→BULL, CHOP→BEAR, any→TRANSITION, etc.)
      held for grace_bars → exit_mandate = True.

    Immediate triggers (no grace period):
      - structural_break = True (CPD fired)
      - Hurst jumps above trending_threshold
      - Volatility regime → EXPANDING

    Parameters
    ----------
    current_consensus : ConsensusState
    previous_consensus : ConsensusState or None (first tick)
    shift_counter : int — how many bars the shift has persisted
    grace_bars : int — required confirmation bars
    exit_cfg : exit_mandate section from config
    hurst_value : float or None
    structural_break : bool
    vol_regime : VolatilityRegime
    hurst_cfg : dict

    Returns
    -------
    (exit_mandate: bool, updated_shift_counter: int)
    """
    if not exit_cfg.get("enabled", True):
        return False, 0

    trending_thresh = hurst_cfg.get("trending_threshold", 0.60)
    triggers = exit_cfg.get("triggers", [])

    # --- Immediate triggers (bypass grace period) ---
    if "cpd_structural_break" in triggers and structural_break:
        logger.info("EXIT MANDATE: structural break detected (immediate)")
        return True, grace_bars  # max out counter

    if "hurst_above_trending" in triggers and hurst_value is not None:
        if hurst_value >= trending_thresh:
            # Only trigger if we were previously in CHOP (range/scalp territory)
            if previous_consensus == ConsensusState.CHOP_NEUTRAL:
                logger.info(
                    "EXIT MANDATE: Hurst %.3f >= %.3f (immediate, was CHOP)",
                    hurst_value, trending_thresh,
                )
                return True, grace_bars

    if "volatility_expanding" in triggers:
        if vol_regime == VolatilityRegime.EXPANDING:
            # Only fire mandate if EXPANDING vol actually changed the consensus state.
            # If consensus stays CHOP despite EXPANDING vol (because Hurst is still
            # sub-trending), this is normal crypto volatility — not an emergency.
            if (previous_consensus == ConsensusState.CHOP_NEUTRAL
                    and current_consensus != ConsensusState.CHOP_NEUTRAL):
                logger.info("EXIT MANDATE: volatility EXPANDING + consensus shifted from CHOP")
                return True, grace_bars

    # --- Grace-period triggers (consensus state change) ---
    if previous_consensus is None:
        # First tick — no shift possible
        return False, 0

    if current_consensus == previous_consensus:
        # No shift — reset counter
        return False, 0

    # Consensus changed
    if "hmm_state_change" in triggers:
        new_counter = shift_counter + 1
        if new_counter >= grace_bars:
            logger.info(
                "EXIT MANDATE: consensus shifted %s → %s (confirmed %d/%d bars)",
                previous_consensus.value, current_consensus.value,
                new_counter, grace_bars,
            )
            return True, new_counter
        else:
            logger.debug(
                "Regime shift detected %s → %s (%d/%d bars, awaiting confirmation)",
                previous_consensus.value, current_consensus.value,
                new_counter, grace_bars,
            )
            return False, new_counter

    # No matching trigger
    return False, 0
