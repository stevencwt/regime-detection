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

def _fit_hmm_with_fallback(X, n_states, cov_type, n_iter, random_state):
    """Fit GaussianHMM with automatic fallback to fewer states.

    Root cause: when data doesn't have enough structure for N states,
    HMM assigns zero observations to one or more states. This causes:
      - transmat_ rows summing to 0 (no transitions from empty state)
      - predict_proba() producing NaN/inf
      - subsequent operations crashing

    Fix: after fitting, check if all states have observations.
    If not, retry with n_states-1 (down to minimum of 2).

    Returns (model, states, posteriors) or (None, None, None) on failure.
    """
    for n in range(n_states, 1, -1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GaussianHMM(
                    n_components=n,
                    covariance_type=cov_type,
                    n_iter=n_iter,
                    random_state=random_state,
                )
                model.fit(X)

                # Check: did all states get observations?
                states = model.predict(X)
                state_counts = np.bincount(states, minlength=n)
                empty_states = int(np.sum(state_counts == 0))

                if empty_states > 0 and n > 2:
                    # 3+ states with empties → retry with fewer
                    logger.debug(
                        "HMM %d-state has %d empty states (counts=%s), retrying with %d",
                        n, empty_states, state_counts.tolist(), n - 1,
                    )
                    continue

                # For 2-state: one empty state is fine — means single regime.
                # Posteriors may have NaN for the empty state, but the populated
                # state's posterior is valid (1.0 for all bars).

                # Safe posteriors: avoid predict_proba if empty states exist
                if empty_states > 0:
                    # Build synthetic posteriors: 1.0 for populated state, 0.0 for empty
                    posteriors = np.zeros((len(states), n))
                    for i, s in enumerate(states):
                        posteriors[i, s] = 1.0
                else:
                    posteriors = model.predict_proba(X)
                    if np.any(np.isnan(posteriors)) or np.any(np.isinf(posteriors)):
                        logger.debug("HMM %d-state posteriors contain NaN/inf, retrying with %d", n, n - 1)
                        continue

                if n < n_states:
                    logger.debug("HMM fell back to %d states (from %d)", n, n_states)
                return model, states, posteriors

        except Exception as e:
            logger.debug("HMM %d-state fit failed (%s), retrying with %d", n, e, n - 1)
            continue

    logger.debug("HMM fitting failed for all state counts (%d down to 2)", n_states)
    return None, None, None


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
    # When log-returns have near-zero variance (identical consecutive prices,
    # stale BBO data), HMM fitting is meaningless. This catches genuinely
    # flat data, NOT the SOL/ETH HTF issue which is caused by empty HMM states.
    return_std = np.std(log_returns)
    if return_std < 1e-10:
        logger.debug("HMM: near-zero return variance (%.2e) — defaulting to CHOP", return_std)
        return "CHOP", np.array([]), 0.5

    X = log_returns.reshape(-1, 1)

    try:
        # Suppress hmmlearn's logger during fitting (convergence warnings).
        import logging as _logging
        _hmm_logger = _logging.getLogger("hmmlearn.base")
        _prev_level = _hmm_logger.level
        _hmm_logger.setLevel(_logging.ERROR)
        try:
            model, states, posteriors = _fit_hmm_with_fallback(
                X, n_states, cov_type, n_iter, random_state
            )
        finally:
            _hmm_logger.setLevel(_prev_level)

        if model is None:
            return "UNKNOWN", np.array([]), 0.0

    except Exception as e:
        logger.debug("HMM fitting failed: %s", e)
        return "UNKNOWN", np.array([]), 0.0

    # --- Auto-label states by mean return ---
    means = model.means_.flatten()  # shape (n_states,)

    # --- Guard: empty states from fallback ---
    # When 2-state HMM has one empty state, the unpopulated state's mean is
    # random noise from initialization. Skip ranking-based labeling and use
    # the populated state's mean sign directly.
    state_counts = np.bincount(states, minlength=len(means))
    if np.any(state_counts == 0):
        # Only consider means of populated states
        populated_mask = state_counts > 0
        populated_means = means[populated_mask]
        avg_mean = np.mean(populated_means)
        if avg_mean > return_std * 0.3:
            fallback_label = "BULL"
        elif avg_mean < -return_std * 0.3:
            fallback_label = "BEAR"
        else:
            fallback_label = "CHOP"
        logger.debug(
            "HMM: empty states detected (counts=%s), populated mean=%.2e → %s",
            state_counts.tolist(), avg_mean, fallback_label,
        )
        return fallback_label, states, float(posteriors[-1, states[-1]])

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


# ===================================================================
# 7. DRIFT DIRECTION — Directional Bias within CHOP Regimes
# ===================================================================

def compute_drift(
    closes: np.ndarray,
    cfg: Dict,
) -> str:
    """Detect directional drift within a CHOP_NEUTRAL regime.

    Distinguishes three sub-states that CHOP_NEUTRAL lumps together:

      UP   — choppy uptrend: price oscillates with upward bias
             (higher highs/lows, price consistently above rising SMA)
      DOWN — choppy downtrend: price oscillates with downward bias
             (lower highs/lows, price consistently below falling SMA)
      NONE — true range-bound: price oscillates around flat SMA
             (no directional bias, classic rectangle/channel)

    Two detection methods (either can trigger):

    Method 1 — SMA-based (slow, stable, catches sustained drift):
      1. Compute SMA(sma_period) over close prices
      2. Compute SMA slope: % change over last slope_window bars
      3. Compute price-above-SMA: fraction of last sma_period bars where close > SMA
      4. Classify using two-tier thresholds:
         - Strong: above_pct >= 0.80 → UP (or <= 0.20 → DOWN)
         - Moderate: above_pct >= 0.65 AND slope confirms → UP/DOWN

    Method 2 — Swing structure (fast, catches developing trends):
      1. Identify local swing highs and lows over a short window
      2. Count consecutive higher-lows (HH/HL) or lower-highs (LH/LL)
      3. If min_swing_count consecutive swings confirm direction → drift detected

    Parameters
    ----------
    closes : 1-D array of close prices (oldest → newest)
    cfg : drift config section with thresholds

    Returns
    -------
    str : "UP", "DOWN", or "NONE"
    """
    sma_period = cfg.get("sma_period", 50)
    slope_window = cfg.get("slope_window", 10)
    strong_above_pct = cfg.get("strong_above_pct", 0.80)
    moderate_above_pct = cfg.get("moderate_above_pct", 0.65)
    min_slope_pct = cfg.get("min_slope_pct", 0.15)
    swing_lookback = cfg.get("swing_lookback", 20)
    swing_order = cfg.get("swing_order", 3)
    min_swing_count = cfg.get("min_swing_count", 2)

    if len(closes) < max(sma_period + slope_window, swing_lookback + 10):
        return "NONE"

    # ===================================================================
    # Method 1: SMA-based (slow, stable)
    # ===================================================================
    sma = np.convolve(closes, np.ones(sma_period) / sma_period, mode='valid')
    sma_result = "NONE"
    if len(sma) >= slope_window + 1:
        sma_now = sma[-1]
        sma_back = sma[-slope_window]
        if sma_back > 0:
            slope_pct = (sma_now - sma_back) / sma_back * 100.0

            n_sma = len(sma)
            aligned_closes = closes[-n_sma:]
            lookback = min(sma_period, n_sma)
            recent_closes = aligned_closes[-lookback:]
            recent_sma = sma[-lookback:]
            above_count = int(np.sum(recent_closes > recent_sma))
            above_pct = above_count / lookback

            if above_pct >= strong_above_pct:
                sma_result = "UP"
            elif above_pct <= (1.0 - strong_above_pct):
                sma_result = "DOWN"
            elif above_pct >= moderate_above_pct and slope_pct > min_slope_pct:
                sma_result = "UP"
            elif above_pct <= (1.0 - moderate_above_pct) and slope_pct < -min_slope_pct:
                sma_result = "DOWN"

    if sma_result != "NONE":
        return sma_result

    # ===================================================================
    # Method 2: Swing structure (fast, catches developing trends)
    # ===================================================================
    # Find local swing highs and lows in the recent window.
    # A swing low is a bar where close is lower than `swing_order` bars
    # on each side. A swing high is the opposite.
    recent = closes[-swing_lookback:]
    swing_lows = []
    swing_highs = []
    for i in range(swing_order, len(recent) - swing_order):
        window_left = recent[i - swing_order:i]
        window_right = recent[i + 1:i + 1 + swing_order]
        # Swing low: lower than all neighbors
        if recent[i] <= np.min(window_left) and recent[i] <= np.min(window_right):
            swing_lows.append((i, recent[i]))
        # Swing high: higher than all neighbors
        if recent[i] >= np.max(window_left) and recent[i] >= np.max(window_right):
            swing_highs.append((i, recent[i]))

    # Count consecutive higher lows (bullish structure)
    higher_lows = 0
    if len(swing_lows) >= 2:
        for j in range(1, len(swing_lows)):
            if swing_lows[j][1] > swing_lows[j - 1][1]:
                higher_lows += 1
            else:
                higher_lows = 0  # reset on break

    # Count consecutive lower highs (bearish structure)
    lower_highs = 0
    if len(swing_highs) >= 2:
        for j in range(1, len(swing_highs)):
            if swing_highs[j][1] < swing_highs[j - 1][1]:
                lower_highs += 1
            else:
                lower_highs = 0  # reset on break

    if higher_lows >= min_swing_count:
        return "UP"
    elif lower_highs >= min_swing_count:
        return "DOWN"

    return "NONE"
