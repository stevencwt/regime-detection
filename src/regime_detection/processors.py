"""
regime_detection.processors — Market-specific processors & consensus voting (Phase 3).

Pure functions that:
  1. Enrich regime output with market-specific context (crypto, options, pairs)
  2. Vote on consensus state from all computed signals

All functions accept raw data + config, return typed dataclasses or enums.
No I/O, no exchange connectors.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from regime_detection.schema import (
    ConsensusState,
    CryptoContext,
    FundingBias,
    HMMLabel,
    LiquidityStatus,
    OptionsContext,
    PairsContext,
    VolatilityRegime,
)
from regime_detection.signals import (
    classify_funding_bias,
    compute_hurst_dfa,
)

logger = logging.getLogger(__name__)


# ===================================================================
# 1. CRYPTO PROCESSOR
# ===================================================================

def process_crypto(
    funding_rate: Optional[float],
    order_book_imbalance: Optional[float],
    funding_cfg: Dict,
) -> CryptoContext:
    """Build CryptoContext from funding rate and order-book data.

    Handles missing data gracefully — any None input produces UNKNOWN
    for that specific field while still computing the others.

    Parameters
    ----------
    funding_rate : float or None
        Current perpetual funding rate (e.g. 0.0003).
    order_book_imbalance : float or None
        Order-book imbalance in [-1, +1].
    funding_cfg : dict
        Funding section from config.

    Returns
    -------
    CryptoContext with populated fields (UNKNOWN for missing inputs).
    """
    bias_str = classify_funding_bias(funding_rate, funding_cfg)

    return CryptoContext(
        funding_bias=bias_str,
        funding_rate=funding_rate,
        order_book_imbalance=order_book_imbalance,
    )


# ===================================================================
# 2. OPTIONS PROCESSOR (US Stocks)
# ===================================================================

def process_options(
    options_snapshot: Optional[Dict[str, Any]],
    options_cfg: Dict,
) -> Optional[OptionsContext]:
    """Build OptionsContext from options greeks / OI snapshot.

    Expected keys in options_snapshot:
      - "vanna" : float        — net vanna signal
      - "gamma" : float        — net gamma exposure
      - "gamma_flip" : float   — gamma flip price level (optional)
      - "oi_skew" : float      — call/put OI skew ratio (optional)

    Parameters
    ----------
    options_snapshot : dict or None
        If None, returns OptionsContext with all-None fields.
    options_cfg : dict
        Options section from config.

    Returns
    -------
    OptionsContext
    """
    if options_snapshot is None:
        return OptionsContext()

    # --- Vanna signal ---
    raw_vanna = options_snapshot.get("vanna")
    vanna_signal = None
    if raw_vanna is not None:
        try:
            vanna_signal = float(raw_vanna)
        except (TypeError, ValueError):
            logger.warning("Invalid vanna value: %s", raw_vanna)

    # --- Gamma boundary ---
    # gamma_flip is the price level where dealer gamma flips from positive to negative
    gamma_boundary = None
    raw_gamma_flip = options_snapshot.get("gamma_flip")
    if raw_gamma_flip is not None:
        try:
            gamma_boundary = float(raw_gamma_flip)
        except (TypeError, ValueError):
            logger.warning("Invalid gamma_flip value: %s", raw_gamma_flip)

    # If gamma_flip not provided, try to infer from net gamma sign
    if gamma_boundary is None and options_cfg.get("gamma_flip_enabled", True):
        raw_gamma = options_snapshot.get("gamma")
        if raw_gamma is not None:
            try:
                # Positive gamma = dealers hedging suppresses vol
                # Negative gamma = dealers amplify moves
                gamma_val = float(raw_gamma)
                # We store the gamma value itself as boundary indicator
                # (positive = supportive, negative = destabilizing)
                gamma_boundary = gamma_val
            except (TypeError, ValueError):
                pass

    # --- OI skew ---
    oi_skew = None
    raw_oi = options_snapshot.get("oi_skew")
    if raw_oi is not None:
        try:
            oi_skew = float(raw_oi)
        except (TypeError, ValueError):
            logger.warning("Invalid oi_skew value: %s", raw_oi)

    return OptionsContext(
        vanna_signal=vanna_signal,
        gamma_boundary=gamma_boundary,
        oi_skew=oi_skew,
    )


# ===================================================================
# 3. PAIRS TRADING PROCESSOR
# ===================================================================

def process_pairs(
    spread_data: Optional[Dict[str, Any]],
    pairs_cfg: Dict,
    hurst_cfg: Dict,
) -> Optional[PairsContext]:
    """Build PairsContext from spread series data.

    Expected keys in spread_data:
      - "spread_series" : list[float] or np.ndarray — historical spread values
      - "half_life" : float (optional) — pre-computed half-life
      - "cointegration_pvalue" : float (optional) — ADF test p-value

    If spread_series is provided, computes spread Hurst via DFA.
    If half_life or cointegration_pvalue are missing, leaves them as None.

    Parameters
    ----------
    spread_data : dict or None
    pairs_cfg : dict — pairs section from config
    hurst_cfg : dict — hurst section from config (for DFA params)

    Returns
    -------
    PairsContext
    """
    if spread_data is None:
        return PairsContext()

    # --- Spread Hurst exponent ---
    spread_hurst = None
    raw_spread = spread_data.get("spread_series")
    if raw_spread is not None:
        try:
            spread_arr = np.asarray(raw_spread, dtype=float)
            if len(spread_arr) >= 50:
                # Use absolute spread values for DFA (treat as price-like series)
                # Shift to ensure all positive for log-return computation
                shifted = spread_arr - np.min(spread_arr) + 1.0
                spread_hurst = round(compute_hurst_dfa(shifted, hurst_cfg), 4)
            else:
                logger.debug("Spread series too short for DFA: %d points", len(spread_arr))
        except Exception as e:
            logger.warning("Spread Hurst computation failed: %s", e)

    # --- Half-life ---
    half_life = None
    raw_hl = spread_data.get("half_life")
    if raw_hl is not None:
        try:
            half_life = float(raw_hl)
        except (TypeError, ValueError):
            logger.warning("Invalid half_life value: %s", raw_hl)

    # --- Cointegration p-value ---
    coint_pvalue = None
    raw_coint = spread_data.get("cointegration_pvalue")
    if raw_coint is not None:
        try:
            coint_pvalue = float(raw_coint)
        except (TypeError, ValueError):
            logger.warning("Invalid cointegration_pvalue: %s", raw_coint)

    return PairsContext(
        spread_hurst=spread_hurst,
        spread_half_life=half_life,
        cointegration_pvalue=coint_pvalue,
    )


# ===================================================================
# 4. CONSENSUS VOTING ENGINE
# ===================================================================

def vote_consensus(
    hmm_label: HMMLabel,
    hurst_value: Optional[float],
    structural_break: bool,
    vol_regime: VolatilityRegime,
    liquidity: LiquidityStatus,
    hurst_cfg: Dict,
    hmm_confidence: float = 0.0,
) -> Tuple[ConsensusState, float]:
    """Determine consensus regime state from all computed signals.

    Voting rules (v3.1 spec Section 4):

    BULL_PERSISTENT:
      - HMM = BULL
      - Hurst >= trending_threshold (persistent trend)
      - No structural break
      - Volatility not EXPANDING (sustained trend, not a spike)

    BEAR_PERSISTENT:
      - HMM = BEAR
      - Hurst >= trending_threshold
      - No structural break
      - Volatility not EXPANDING

    CHOP_NEUTRAL:
      - HMM = CHOP (or any label with Hurst below trending)
      - Hurst < trending_threshold
      - No structural break

    TRANSITION:
      - Structural break detected, OR
      - Signals conflict (e.g. HMM=BULL but Hurst is mean-reverting), OR
      - Volatility EXPANDING with structural break

    UNKNOWN:
      - Insufficient data for any signal

    Parameters
    ----------
    hmm_label : HMMLabel
    hurst_value : float or None
    structural_break : bool
    vol_regime : VolatilityRegime
    liquidity : LiquidityStatus
    hurst_cfg : dict — hurst section from config (for thresholds)
    hmm_confidence : float — posterior probability from HMM

    Returns
    -------
    (ConsensusState, confidence_score)
    """
    trending_thresh = hurst_cfg.get("trending_threshold", 0.60)
    mr_thresh = hurst_cfg.get("mean_reverting_threshold", 0.40)

    # --- Handle insufficient data ---
    if hmm_label == HMMLabel.UNKNOWN or hurst_value is None:
        return ConsensusState.UNKNOWN, 0.0

    # --- Structural break overrides toward TRANSITION ---
    if structural_break:
        # CPD fired recently — market is shifting
        confidence = _compute_confidence(
            hmm_confidence, hurst_value, trending_thresh, structural_break
        )
        return ConsensusState.TRANSITION, confidence

    # --- Volatility expanding + extreme Hurst = TRANSITION ---
    if vol_regime == VolatilityRegime.EXPANDING and hurst_value >= trending_thresh:
        # Could be a blowoff or crash — transitional state
        confidence = _compute_confidence(
            hmm_confidence, hurst_value, trending_thresh, structural_break
        )
        return ConsensusState.TRANSITION, confidence

    # --- BULL_PERSISTENT ---
    if hmm_label == HMMLabel.BULL and hurst_value >= trending_thresh:
        confidence = _compute_confidence(
            hmm_confidence, hurst_value, trending_thresh, structural_break
        )
        return ConsensusState.BULL_PERSISTENT, confidence

    # --- BEAR_PERSISTENT ---
    if hmm_label == HMMLabel.BEAR and hurst_value >= trending_thresh:
        confidence = _compute_confidence(
            hmm_confidence, hurst_value, trending_thresh, structural_break
        )
        return ConsensusState.BEAR_PERSISTENT, confidence

    # --- CHOP_NEUTRAL ---
    # HMM=CHOP with sub-trending Hurst, OR
    # HMM=BULL/BEAR but Hurst doesn't confirm persistence
    if hurst_value < trending_thresh:
        confidence = _compute_confidence(
            hmm_confidence, hurst_value, trending_thresh, structural_break
        )
        return ConsensusState.CHOP_NEUTRAL, confidence

    # --- Signal conflict → TRANSITION ---
    # e.g. HMM says CHOP but Hurst says trending
    confidence = _compute_confidence(
        hmm_confidence, hurst_value, trending_thresh, structural_break
    )
    return ConsensusState.TRANSITION, confidence


def _compute_confidence(
    hmm_confidence: float,
    hurst_value: float,
    trending_thresh: float,
    structural_break: bool,
) -> float:
    """Compute an aggregate confidence score [0, 1].

    Blends:
      - HMM posterior probability (40% weight)
      - Hurst clarity — distance from 0.5 random walk (30% weight)
      - Stability bonus — no structural break (20% weight)
      - Base confidence floor (10%)

    Returns
    -------
    float in [0, 1]
    """
    # HMM component: how sure is the HMM of its current state
    hmm_score = float(np.clip(hmm_confidence, 0.0, 1.0))

    # Hurst clarity: how far from 0.5 (higher distance = more conviction)
    hurst_distance = abs(hurst_value - 0.5)
    hurst_score = float(np.clip(hurst_distance / 0.3, 0.0, 1.0))  # normalize

    # Stability: no break = bonus
    stability_score = 0.0 if structural_break else 1.0

    # Weighted sum
    confidence = (
        0.40 * hmm_score
        + 0.30 * hurst_score
        + 0.20 * stability_score
        + 0.10  # floor
    )

    return float(np.clip(confidence, 0.0, 1.0))
