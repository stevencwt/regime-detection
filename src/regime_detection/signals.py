"""
regime_detection.signals — Core signal computation functions (Phase 2).

All functions in this module are PURE — they take numpy arrays or scalars
and return results.  No exchange connectors, no I/O, no state.

Functions:
    compute_hurst_dfa     — DFA-based Hurst exponent (fathon or numpy fallback)
    compute_hmm_labels    — GaussianHMM fit → BULL/BEAR/CHOP labeling
    compute_cpd           — BinSeg change-point detection
    classify_volatility   — Rolling vol → LOW_STABLE / MODERATE / EXPANDING / CONTRACTING
    classify_liquidity    — Order-book imbalance → CONSOLIDATION / TRAP / PASSED
    classify_funding_bias — Funding rate → EXTREME_POSITIVE / NEGATIVE / NEUTRAL
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports with graceful fallback
# ---------------------------------------------------------------------------

_FATHON_AVAILABLE = False
try:
    import fathon
    from fathon import fathonUtils as fu
    _FATHON_AVAILABLE = True
except ImportError:
    pass

_HMMLEARN_AVAILABLE = False
try:
    from hmmlearn.hmm import GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:
    pass

_RUPTURES_AVAILABLE = False
try:
    import ruptures
    _RUPTURES_AVAILABLE = True
except ImportError:
    pass


# ===================================================================
# 1. DFA HURST EXPONENT
# ===================================================================

def _dfa_hurst_numpy(series: np.ndarray, min_win: int = 10,
                     max_win_ratio: float = 0.25, order: int = 1) -> float:
    """Pure-numpy DFA Hurst exponent (fallback when fathon unavailable).

    Detrended Fluctuation Analysis:
      1. Compute cumulative deviation from mean (profile).
      2. For each window size, split profile into segments, fit polynomial
         of given order, compute RMS of residuals.
      3. Log-log regression of RMS vs window size → slope = Hurst exponent.

    Parameters
    ----------
    series : 1-D array of float (close prices or log-returns)
    min_win : minimum window size
    max_win_ratio : max window = floor(len * ratio)
    order : detrending polynomial order (1 = linear)

    Returns
    -------
    float : Hurst exponent estimate (0→mean-reverting, 0.5→random, 1→trending)
    """
    n = len(series)
    max_win = max(min_win + 1, int(n * max_win_ratio))

    if max_win <= min_win or n < min_win * 4:
        return 0.5  # insufficient data

    # Profile: cumulative sum of deviations from mean
    profile = np.cumsum(series - np.mean(series))

    # Window sizes: logarithmically spaced integers
    win_sizes = np.unique(
        np.logspace(np.log10(min_win), np.log10(max_win), num=20).astype(int)
    )
    win_sizes = win_sizes[(win_sizes >= min_win) & (win_sizes <= max_win)]

    if len(win_sizes) < 4:
        return 0.5

    fluctuations = []
    valid_wins = []

    for w in win_sizes:
        n_segments = n // w
        if n_segments < 1:
            continue

        rms_values = []
        for seg_idx in range(n_segments):
            start = seg_idx * w
            end = start + w
            segment = profile[start:end]

            # Detrend with polynomial fit
            x = np.arange(w, dtype=float)
            coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(coeffs, x)
            residuals = segment - trend

            rms = np.sqrt(np.mean(residuals ** 2))
            rms_values.append(rms)

        if rms_values:
            fluctuations.append(np.mean(rms_values))
            valid_wins.append(w)

    if len(valid_wins) < 4:
        return 0.5

    # Log-log regression
    log_wins = np.log(np.array(valid_wins, dtype=float))
    log_fluct = np.log(np.array(fluctuations, dtype=float))

    # Filter out any inf/nan
    mask = np.isfinite(log_wins) & np.isfinite(log_fluct)
    if mask.sum() < 4:
        return 0.5

    slope, _ = np.polyfit(log_wins[mask], log_fluct[mask], 1)
    return float(np.clip(slope, 0.0, 1.5))


def _dfa_hurst_fathon(series: np.ndarray, min_win: int = 10,
                      max_win_ratio: float = 0.25, order: int = 1) -> float:
    """DFA Hurst via fathon library (faster C implementation)."""
    n = len(series)
    max_win = max(min_win + 2, int(n * max_win_ratio))

    if max_win <= min_win or n < min_win * 4:
        return 0.5

    try:
        # fathon expects the cumulative sum (profile) as input
        profile = fu.toAggregated(series)
        dfa_obj = fathon.DFA(profile)

        win_sizes = fu.linRangeByStep(min_win, max_win, step=max(1, (max_win - min_win) // 20))

        _, fluct = dfa_obj.computeFlucVec(win_sizes, polOrd=order)
        hurst, _ = dfa_obj.fitFlucVec()
        return float(np.clip(hurst, 0.0, 1.5))
    except Exception as e:
        logger.warning("fathon DFA failed (%s), falling back to numpy", e)
        return _dfa_hurst_numpy(series, min_win, max_win_ratio, order)


def compute_hurst_dfa(closes: np.ndarray, cfg: Dict) -> float:
    """Compute DFA Hurst exponent from close prices.

    Parameters
    ----------
    closes : 1-D array of close prices (oldest → newest)
    cfg : hurst section of config dict

    Returns
    -------
    float : Hurst exponent
    """
    if len(closes) < 30:
        return 0.5

    min_win = cfg.get("min_window", 10)
    max_win_ratio = cfg.get("max_window_ratio", 0.25)
    order = cfg.get("order", 1)

    # Use log-returns for stationarity
    log_returns = np.diff(np.log(closes[closes > 0]))
    if len(log_returns) < min_win * 4:
        return 0.5

    if _FATHON_AVAILABLE:
        return _dfa_hurst_fathon(log_returns, min_win, max_win_ratio, order)
    else:
        return _dfa_hurst_numpy(log_returns, min_win, max_win_ratio, order)


# ===================================================================
# 2. GAUSSIAN HMM — BULL / BEAR / CHOP LABELING
# ===================================================================

def compute_hmm_labels(
    closes: np.ndarray,
    cfg: Dict,
    stability_bars: int = 2,
) -> Tuple[str, np.ndarray, float]:
    """Fit GaussianHMM to log-returns and label states as BULL/BEAR/CHOP.

    Auto-labeling logic:
      - State with highest mean return → BULL
      - State with lowest mean return  → BEAR
      - Middle state(s)                → CHOP

    Robustness features:
      - Near-zero variance detection → returns CHOP (market is flat/stale)
      - Means-separation check → if all state means are nearly equal, returns CHOP
      - Majority vote stability → uses most-common label over last N bars,
        not unanimity (prevents oscillation-induced UNKNOWN)

    Parameters
    ----------
    closes : 1-D array of close prices
    cfg : hmm section of config dict
    stability_bars : window size for majority-vote stability check

    Returns
    -------
    (current_label, state_sequence, confidence)
        current_label : "BULL" | "BEAR" | "CHOP" | "UNKNOWN"
        state_sequence : array of state indices
        confidence : posterior probability of current state
    """
    if not _HMMLEARN_AVAILABLE:
        logger.warning("hmmlearn not available — HMM disabled")
        return "UNKNOWN", np.array([]), 0.0

    min_bars = cfg.get("min_training_bars", 100)
    if len(closes) < min_bars:
        return "UNKNOWN", np.array([]), 0.0

    n_states = cfg.get("n_states", 3)
    cov_type = cfg.get("covariance_type", "full")
    n_iter = cfg.get("n_iter", 50)
    random_state = cfg.get("random_state", 42)

    # Log-returns as observable
    log_returns = np.diff(np.log(closes[closes > 0]))
    if len(log_returns) < min_bars:
        return "UNKNOWN", np.array([]), 0.0

    # --- Guard: near-zero variance (stale/flat data) ---
    # When consecutive prices are identical (e.g., BBO polling), log-returns
    # are ~0.0 and the HMM cannot differentiate states meaningfully.
    return_std = np.std(log_returns)
    if return_std < 1e-10:
        logger.debug("HMM: near-zero return variance (%.2e) — defaulting to CHOP", return_std)
        return "CHOP", np.array([]), 0.5

    X = log_returns.reshape(-1, 1)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=cov_type,
                n_iter=n_iter,
                random_state=random_state,
            )
            model.fit(X)
            states = model.predict(X)
            posteriors = model.predict_proba(X)

    except Exception as e:
        logger.warning("HMM fitting failed: %s", e)
        return "UNKNOWN", np.array([]), 0.0

    # --- Auto-label states by mean return ---
    means = model.means_.flatten()  # shape (n_states,)

    # --- Guard: means separation check ---
    # If all state means are nearly identical, the HMM cannot distinguish
    # regimes — but the overall sign tells us direction.
    means_range = np.max(means) - np.min(means)
    if means_range < return_std * 0.1:
        # Means are clustered — single regime in the data.
        # Use the sign of the average mean to determine direction.
        avg_mean = np.mean(means)
        if avg_mean > return_std * 0.3:
            fallback_label = "BULL"
        elif avg_mean < -return_std * 0.3:
            fallback_label = "BEAR"
        else:
            fallback_label = "CHOP"

        logger.debug(
            "HMM: state means not separated (range=%.2e, avg=%.2e) — using %s",
            means_range, avg_mean, fallback_label,
        )
        return fallback_label, states, float(posteriors[-1, states[-1]])

    ranked = np.argsort(means)      # lowest → highest mean

    label_map = {}
    label_map[ranked[-1]] = "BULL"   # highest mean → BULL
    label_map[ranked[0]] = "BEAR"    # lowest mean → BEAR
    for idx in ranked[1:-1]:
        label_map[idx] = "CHOP"      # middle → CHOP

    # Map state sequence to labels
    labeled_sequence = np.array([label_map[s] for s in states])

    # --- Stability check: majority vote over recent window ---
    # Uses the most-common label in the last N bars instead of requiring
    # all N bars to agree. This prevents oscillation-induced UNKNOWN.
    vote_window = max(stability_bars, 5)  # minimum 5 bars for meaningful vote
    if len(labeled_sequence) >= vote_window:
        recent = labeled_sequence[-vote_window:]
        # Count occurrences of each label
        labels, counts = np.unique(recent, return_counts=True)
        majority_idx = np.argmax(counts)
        majority_label = labels[majority_idx]
        majority_pct = counts[majority_idx] / len(recent)

        # Require >50% majority for a definitive label
        if majority_pct > 0.5:
            current_label = majority_label
        else:
            # No clear majority — market is transitioning
            current_label = "CHOP"  # default to CHOP when ambiguous
    else:
        current_label = labeled_sequence[-1]

    # Confidence = posterior probability of current state
    current_state_idx = states[-1]
    confidence = float(posteriors[-1, current_state_idx])

    return current_label, states, confidence


# ===================================================================
# 3. CHANGE POINT DETECTION (BinSeg + RBF kernel)
# ===================================================================

def compute_cpd(
    closes: np.ndarray,
    cfg: Dict,
) -> Tuple[bool, List[int]]:
    """Detect structural breaks via BinSeg change-point detection.

    Parameters
    ----------
    closes : 1-D array of close prices
    cfg : cpd section of config dict

    Returns
    -------
    (structural_break, breakpoints)
        structural_break : True if a breakpoint occurred within last `recency_bars`
        breakpoints : list of breakpoint indices (in the lookback window)
    """
    if not _RUPTURES_AVAILABLE:
        logger.warning("ruptures not available — CPD disabled")
        return False, []

    lookback = cfg.get("lookback_bars", 200)
    model_name = cfg.get("model", "rbf")
    penalty = cfg.get("penalty", 5.0)
    min_size = cfg.get("min_size", 20)
    recency = cfg.get("recency_bars", 5)

    if len(closes) < lookback:
        # Use whatever we have if it's enough for min_size
        if len(closes) < min_size * 2:
            return False, []
        window = closes
    else:
        window = closes[-lookback:]

    # Use log-returns for stationarity
    log_returns = np.diff(np.log(window[window > 0]))
    if len(log_returns) < min_size * 2:
        return False, []

    try:
        algo = ruptures.Binseg(model=model_name, min_size=min_size).fit(
            log_returns.reshape(-1, 1)
        )
        breakpoints = algo.predict(pen=penalty)

        # ruptures appends len(signal) as last element — remove it
        breakpoints = [bp for bp in breakpoints if bp < len(log_returns)]

        # Check if any breakpoint is within the most recent N bars
        if breakpoints:
            most_recent_bp = max(breakpoints)
            recency_threshold = len(log_returns) - recency
            structural_break = most_recent_bp >= recency_threshold
        else:
            structural_break = False

        return structural_break, breakpoints

    except Exception as e:
        logger.warning("CPD computation failed: %s", e)
        return False, []


# ===================================================================
# 4. VOLATILITY REGIME CLASSIFICATION
# ===================================================================

def classify_volatility(
    closes: np.ndarray,
    cfg: Dict,
) -> str:
    """Classify current volatility regime.

    Method:
      1. Compute rolling standard deviation of log-returns.
      2. Compare current vol to historical median vol.
      3. Classify based on ratio thresholds.

    Returns one of: LOW_STABLE, MODERATE, EXPANDING, CONTRACTING, UNKNOWN
    """
    window = cfg.get("window", 20)
    expanding_thresh = cfg.get("expanding_threshold", 1.5)
    contracting_thresh = cfg.get("contracting_threshold", 0.6)
    stable_band = cfg.get("stable_band", [0.7, 1.3])
    low_pctile = cfg.get("low_stable_percentile", 40)

    if len(closes) < window * 2:
        return "UNKNOWN"

    log_returns = np.diff(np.log(closes[closes > 0]))
    if len(log_returns) < window * 2:
        return "UNKNOWN"

    # Rolling volatility (std of log-returns)
    vol_series = np.array([
        np.std(log_returns[max(0, i - window):i])
        for i in range(window, len(log_returns) + 1)
    ])

    if len(vol_series) < 2:
        return "UNKNOWN"

    current_vol = vol_series[-1]
    median_vol = np.median(vol_series)

    if median_vol <= 0:
        return "UNKNOWN"

    ratio = current_vol / median_vol

    # Classification
    if ratio >= expanding_thresh:
        return "EXPANDING"
    elif ratio <= contracting_thresh:
        return "CONTRACTING"
    elif stable_band[0] <= ratio <= stable_band[1]:
        # Distinguish LOW_STABLE from MODERATE by percentile rank
        percentile_rank = np.sum(vol_series <= current_vol) / len(vol_series) * 100
        if percentile_rank <= low_pctile:
            return "LOW_STABLE"
        else:
            return "MODERATE"
    else:
        return "MODERATE"


# ===================================================================
# 5. LIQUIDITY HEURISTIC (Order-Book Imbalance)
# ===================================================================

def classify_liquidity(
    order_book_imbalance: Optional[float],
    cfg: Dict,
) -> str:
    """Classify liquidity status from order-book imbalance.

    Parameters
    ----------
    order_book_imbalance : float in [-1, +1] or None
    cfg : liquidity section of config

    Returns
    -------
    "CONSOLIDATION" | "LIQUIDITY_TRAP" | "PASSED" | "UNKNOWN"
    """
    if order_book_imbalance is None:
        return "UNKNOWN"

    trap_thresh = cfg.get("imbalance_trap_threshold", 0.7)
    consol_band = cfg.get("consolidation_band", [-0.3, 0.3])

    imb = float(order_book_imbalance)

    if abs(imb) >= trap_thresh:
        return "LIQUIDITY_TRAP"
    elif consol_band[0] <= imb <= consol_band[1]:
        return "CONSOLIDATION"
    else:
        return "PASSED"


# ===================================================================
# 6. FUNDING RATE BIAS (Crypto-specific)
# ===================================================================

def classify_funding_bias(
    funding_rate: Optional[float],
    cfg: Dict,
) -> str:
    """Classify crypto perpetual funding rate bias.

    Returns "EXTREME_POSITIVE" | "EXTREME_NEGATIVE" | "NEUTRAL" | "UNKNOWN"
    """
    if funding_rate is None:
        return "UNKNOWN"

    extreme_pos = cfg.get("extreme_positive", 0.01)
    extreme_neg = cfg.get("extreme_negative", -0.01)
    neutral_band = cfg.get("neutral_band", [-0.005, 0.005])

    rate = float(funding_rate)

    if rate >= extreme_pos:
        return "EXTREME_POSITIVE"
    elif rate <= extreme_neg:
        return "EXTREME_NEGATIVE"
    elif neutral_band[0] <= rate <= neutral_band[1]:
        return "NEUTRAL"
    else:
        # Between neutral and extreme — still call it neutral
        return "NEUTRAL"
