# Technical Specification: Multi-Modal Market Regime Analysis Framework
# v3.2 — Post-Implementation Update (Reflects Actual Built System)

---

## Changelog from v3.1

- **v3.2**: Updated to reflect actual implementation after 5-phase build + live testing.
  - API signature expanded (order_book_imbalance, spread_data as explicit params)
  - HMM robustness features documented (near-zero variance guard, direction-aware fallback, majority vote)
  - Config key names aligned to implementation
  - Confidence score calculation documented
  - TRANSITION consensus rules detailed
  - `scalping_max_hold_min` removed (downstream bot concern, not regime detection)
  - Temporal matrix simplified to single TF per entry (configurable)
  - Section 8 added: HMM Robustness (learned from live Hyperliquid testing)
  - Section 9 added: Implementation Architecture (package structure)

---

## 1. Executive Summary

This framework serves as the **robust, reusable foundational regime detection layer** for a full trading bot. It classifies the current market state and recommends execution logic, enabling:
- Entry filtering (only allow trades in matching regimes).
- Mandatory exits on regime change (e.g., exit all range/scalp positions when Hurst rises or CPD triggers).
- Downstream modules query the RegimeManager for real-time consensus.

Supported categories (with clear Range vs Scalp split):
- Swing (Crypto/US Stocks)
- **Scalping** (Crypto/US Stocks — ultra-short mean-reversion)
- **Range Trading** (Crypto/US Stocks — longer-hold mean-reversion in clean ranges)
- Options Income/Selling (US Stocks)
- Options Speculative/Buying (US Stocks)
- Pairs Trading (Crypto Perps on Hyperliquid)

**Implementation status**: Fully built, 185 tests passing, validated with live Hyperliquid data.
**Repository**: `https://github.com/stevencwt/regime-detection`

---

## 2. The Hybrid Consensus Engine

The consensus engine combines five independent signals via weighted voting:

| Signal | Library | Output | Weight in Confidence |
|---|---|---|---|
| GaussianHMM (3-state) | hmmlearn | BULL / BEAR / CHOP | 40% (posterior probability) |
| DFA Hurst Exponent | fathon or numpy | 0.0 – 1.0 | 30% (distance from 0.5) |
| BinSeg CPD | ruptures | structural_break bool | 20% (stability bonus) |
| Volatility Regime | numpy (rolling std) | LOW_STABLE / MODERATE / EXPANDING / CONTRACTING | Input to consensus rules |
| Liquidity Heuristic | threshold classifier | CONSOLIDATION / TRAP / PASSED | Input to recommendation rules |

Additional market-specific processors:
- **Crypto**: Funding rate bias (NEUTRAL / EXTREME_POSITIVE / EXTREME_NEGATIVE)
- **US Stocks**: Vanna signal, gamma boundary, OI skew
- **Pairs**: Spread Hurst, half-life, cointegration p-value

---

## 3. Temporal & Lookback Matrix

Each strategy type resolves to a specific set of temporal parameters via the
`temporal_matrix` config section. The `strategy_type` and `market_class` arguments
to `RegimeManager()` select the row.

| Strategy Type        | Market    | Signal TF | Execution TF | Lookback (bars) | HMM Stability (bars) |
|----------------------|-----------|-----------|--------------|-----------------|----------------------|
| scalping             | crypto    | 5m        | 1m           | 750             | 2                    |
| scalping             | us_stocks | 5m        | 1m           | 500             | 3                    |
| range_trading        | crypto    | 15m       | 5m           | 1000            | 3                    |
| range_trading        | us_stocks | 1h        | 15m          | 800             | 4                    |
| swing                | crypto    | 1h        | 15m          | 1000            | 3                    |
| swing                | us_stocks | 1h        | 15m          | 800             | 4                    |
| options_income       | us_stocks | 1d        | 1d           | 252             | 5                    |
| options_speculative  | us_stocks | 1d        | 30m          | 252             | 4                    |
| pairs_trading        | crypto    | 15m       | 5m           | 1000            | 3                    |

**Note**: Signal TF and Execution TF are informational for the downstream bot.
The RegimeManager itself is timeframe-agnostic — it processes whatever bars are fed
via `.update()`. The `lookback_bars` value sets the internal FIFO buffer size.

---

## 4. Regime Definitions & Thresholds

### 4.1 Consensus States

| State | Conditions |
|---|---|
| **BULL_PERSISTENT** | HMM=BULL + Hurst ≥ 0.60 + no structural break + vol not EXPANDING |
| **BEAR_PERSISTENT** | HMM=BEAR + Hurst ≥ 0.60 + no structural break + vol not EXPANDING |
| **CHOP_NEUTRAL** | Hurst < 0.60 (regardless of HMM label — Hurst overrides HMM for persistence) |
| **TRANSITION** | Structural break detected, OR EXPANDING vol + trending Hurst, OR signal conflict (HMM=CHOP but Hurst trending) |
| **UNKNOWN** | Insufficient data (HMM=UNKNOWN or Hurst=None, typically during warmup) |

### 4.2 CHOP_NEUTRAL Sub-Flavors (for recommendation)

| Sub-Regime | Hurst Range | Volatility | Liquidity | Persistence | Recommendation |
|---|---|---|---|---|---|
| Noisy chop | < 0.48 | any | CONSOLIDATION | N/A | SCALP_MEAN_REVERSION |
| Clean range | 0.48 – 0.58 | LOW_STABLE or CONTRACTING | CONSOLIDATION or PASSED | ≥ 10 bars | RANGE_TRADING |
| Developing range | 0.48 – 0.58 | any | CONSOLIDATION or PASSED | < 10 bars | SCALP_MEAN_REVERSION (interim) |
| Trap | any | any | LIQUIDITY_TRAP | N/A | NO_TRADE |

### 4.3 Hurst Thresholds (configurable)

```yaml
hurst:
  trending_threshold: 0.60      # H ≥ 0.60 → persistent trend
  mean_reverting_threshold: 0.40 # H ≤ 0.40 → strong mean-reversion
  range_min_hurst: 0.48          # sweet spot lower bound
  range_max_hurst: 0.58          # sweet spot upper bound
```

### 4.4 Volatility Regime Classification

Method: Rolling std of log-returns (window=20 bars) vs historical median.

| Regime | Condition |
|---|---|
| EXPANDING | current_vol / median_vol ≥ 1.5 |
| CONTRACTING | current_vol / median_vol ≤ 0.6 |
| LOW_STABLE | Within [0.7, 1.3] band AND below 40th percentile |
| MODERATE | Within [0.7, 1.3] band AND above 40th percentile |

---

## 5. Strategy-Specific Regime Activation Logic

### 5.1 Safety Gates (evaluated first, apply to ALL strategies)

| Condition | Result |
|---|---|
| consensus = UNKNOWN | NO_TRADE |
| consensus = TRANSITION | NO_TRADE |
| structural_break = True | NO_TRADE |
| Hurst = None | NO_TRADE |

### 5.2 Strategy-Specific Overrides (evaluated second)

| Strategy Type | Consensus | Result |
|---|---|---|
| pairs_trading | CHOP_NEUTRAL | PAIRS_MEAN_REVERSION |
| pairs_trading | any trending | NO_TRADE |
| options_income | any (stable/contracting vol) | OPTIONS_INCOME |
| options_income | EXPANDING vol | NO_TRADE |
| options_speculative | trending or EXPANDING vol | OPTIONS_SPECULATIVE |
| options_speculative | other | NO_TRADE |

### 5.3 General Strategies (evaluated third)

| Consensus | Result |
|---|---|
| BULL_PERSISTENT | SWING_TREND_FOLLOW |
| BEAR_PERSISTENT | SWING_TREND_FOLLOW |
| CHOP_NEUTRAL | → Sub-regime classification (see Section 4.2) |

### 5.4 Scalping (Ultra-short mean-reversion)

- **Recommended Logic**: SCALP_MEAN_REVERSION
- **Activation**: Consensus=CHOP_NEUTRAL + Hurst < 0.48 + Liquidity=CONSOLIDATION
- **Style**: Many small trades; enter on micro-reversions; target 0.1–0.5%.
- **Exit**: Regime change (exit_mandate) OR trailing stop OR time limit.

### 5.5 Range Trading (Buy bottom / Sell top)

- **Recommended Logic**: RANGE_TRADING
- **Activation** (ALL must be true):
  - Consensus = CHOP_NEUTRAL
  - Hurst ∈ [0.48, 0.58]
  - Volatility = LOW_STABLE or CONTRACTING
  - Liquidity = CONSOLIDATION or PASSED
  - Range persistence ≥ 10 bars (price inside Donchian/Keltner channel)
- **Style**: Fewer trades; wait for channel extremes; target 50–80% of range width.
- **Exit**: Regime change (mandatory) OR opposite extreme OR range breakout.
- **Range hints**: Provided in JSON output with channel boundaries and current deviation.

### 5.6 Global Mandatory Exit Rule

Any regime shift triggers `exit_mandate = true`:

| Trigger | Grace Period |
|---|---|
| CPD structural break | **Immediate** (no grace) |
| Hurst ≥ 0.60 when previously CHOP_NEUTRAL | **Immediate** |
| Volatility → EXPANDING when previously CHOP_NEUTRAL | **Immediate** |
| Consensus state change (e.g. CHOP→BULL) | 2-bar confirmation (configurable) |

When `exit_mandate = true`, all downstream bots must close all open positions
regardless of P&L. The grace period prevents whipsaw exits on noisy state changes.

---

## 6. Standardized Output Schema (JSON)

```json
{
  "consensus_state": "CHOP_NEUTRAL",
  "market_type": "CRYPTO_PERP",
  "confidence_score": 0.72,
  "volatility_regime": "LOW_STABLE",
  "signals": {
    "hmm_label": "CHOP",
    "hurst_dfa": 0.52,
    "structural_break": false,
    "liquidity_status": "CONSOLIDATION",
    "crypto_context": {
      "funding_bias": "NEUTRAL",
      "funding_rate": 0.0003,
      "order_book_imbalance": -0.12
    },
    "range_hints": {
      "is_clean_range": true,
      "range_lower": 145.20,
      "range_upper": 152.80,
      "current_deviation_pct": -1.4,
      "channel_type": "Donchian_30"
    }
  },
  "recommended_logic": "RANGE_TRADING",
  "exit_mandate": false,
  "timestamp": "2026-03-14T14:30:00+00:00"
}
```

### Optional Contexts (appear only when relevant market_type is set):

- `signals.crypto_context` — CRYPTO_PERP / CRYPTO_SPOT
- `signals.options_context` — US_STOCK / US_OPTION (fields: vanna_signal, gamma_boundary, oi_skew)
- `signals.pairs_context` — SPREAD (fields: spread_hurst, spread_half_life, cointegration_pvalue)
- `signals.range_hints` — CHOP_NEUTRAL + SCALP or RANGE recommendation

### Confidence Score Calculation

Weighted blend of signal quality:

| Component | Weight | Meaning |
|---|---|---|
| HMM posterior probability | 40% | How certain HMM is of current state |
| Hurst clarity (distance from 0.5) | 30% | How far from random walk (more distance = more conviction) |
| Stability bonus (no structural break) | 20% | Reward for stable market structure |
| Floor | 10% | Minimum confidence when signals exist |

---

## 7. Public API (Implemented)

```python
from regime_detection import RegimeManager

# Constructor
manager = RegimeManager(
    config_path=None,           # str | Path — YAML override file
    market_type="CRYPTO_PERP",  # CRYPTO_PERP | CRYPTO_SPOT | US_STOCK | US_OPTION | SPREAD
    strategy_type="scalping",   # scalping | range_trading | swing | pairs_trading | options_income | options_speculative
    market_class="crypto",      # crypto | us_stocks
)

# Feed data (every tick)
manager.update(
    bar={"timestamp": ..., "o": ..., "h": ..., "l": ..., "c": ..., "v": ...},
    funding_rate=0.0003,            # float | None — crypto perp funding
    order_book_imbalance=-0.12,     # float [-1,+1] | None — order book bias
    options_snapshot={"vanna": 0.45, "gamma": -0.3, "oi_skew": 1.2},  # dict | None
    spread_data={"spread_series": [...], "half_life": 18.0},           # dict | None
)

# Read output
regime = manager.get_current_regime()   # dict
json_str = manager.get_json()            # JSON string
bars = manager.bar_count                 # int

# Hot-reload config
issues = manager.reload_config()         # list of warnings

# Key downstream checks
if regime["exit_mandate"]:
    # close all positions immediately
    pass
logic = regime["recommended_logic"]      # e.g. "SCALP_MEAN_REVERSION"
```

---

## 8. HMM Robustness Features

Added after live testing with Hyperliquid BBO data, where 5-second polling
produced near-identical consecutive prices that caused the original HMM to
oscillate and stay UNKNOWN indefinitely.

### 8.1 Near-Zero Variance Guard

When `std(log_returns) < 1e-10` (stale/flat data), the HMM is skipped entirely
and returns `CHOP` with 0.5 confidence. This handles:
- BBO polling faster than candle timeframe
- Exchange maintenance periods
- Illiquid assets with no price movement

### 8.2 Direction-Aware Means Fallback

When all 3 HMM state means cluster together (single-regime data where the HMM
cannot differentiate states), the module checks the average of all state means:
- Average strongly positive (> 0.3 × std) → `BULL`
- Average strongly negative (< -0.3 × std) → `BEAR`
- Average near zero → `CHOP`

This correctly labels a pure uptrend as BULL instead of CHOP.

### 8.3 Majority Vote Stability

The stability check uses a **majority vote** over `max(stability_bars, 5)` recent
bars. The most-common label needs >50% to win. If no label has majority, defaults
to `CHOP`. This replaces the earlier unanimity requirement which caused persistent
UNKNOWN during noisy periods.

---

## 9. Implementation Architecture

### 9.1 Package Structure

```
regime-detection/
├── pyproject.toml                      # hatchling, deps: numpy, pandas, hmmlearn, ruptures, scipy
├── config/default_config.yaml          # all thresholds (157 lines)
├── src/regime_detection/
│   ├── __init__.py                     # exports: RegimeManager + enums
│   ├── config.py                       # YAML load → deep merge → validate → hot-reload
│   ├── schema.py                       # enums (ConsensusState, RecommendedLogic, etc.) + dataclasses + JSON
│   ├── signals.py                      # HMM, DFA Hurst, CPD, volatility, liquidity, funding (pure functions)
│   ├── processors.py                   # crypto/options/pairs context builders + consensus voting
│   └── recommendation.py              # recommended logic + range hints + range persistence + exit mandate
└── tests/ (185 tests)
```

### 9.2 Data Flow

```
.update(bar, funding_rate, order_book_imbalance, options_snapshot, spread_data)
    │
    ├─ Validate bar → append to FIFO buffer (maxlen = lookback_bars)
    ├─ Check warmup (< min_training_bars → return UNKNOWN)
    │
    ├─ signals.py:
    │   ├─ compute_hmm_labels()      → HMMLabel + confidence
    │   ├─ compute_hurst_dfa()       → float
    │   ├─ compute_cpd()             → bool + breakpoints
    │   ├─ classify_volatility()     → VolatilityRegime
    │   └─ classify_liquidity()      → LiquidityStatus
    │
    ├─ processors.py:
    │   ├─ process_crypto()          → CryptoContext
    │   ├─ process_options()         → OptionsContext
    │   ├─ process_pairs()           → PairsContext
    │   └─ vote_consensus()          → ConsensusState + confidence
    │
    ├─ recommendation.py:
    │   ├─ compute_range_persistence() → int
    │   ├─ determine_recommended_logic() → RecommendedLogic
    │   ├─ compute_range_hints()     → RangeHints | None
    │   └─ evaluate_exit_mandate()   → bool + counter
    │
    └─ Build RegimeOutput → .get_json() / .get_current_regime()
```

### 9.3 Configuration

All thresholds are in `config/default_config.yaml`. Override any value by
providing a partial YAML file to the constructor:

```python
manager = RegimeManager(config_path="my_overrides.yaml")
```

Key configurable parameters:

```yaml
hmm:
  n_states: 3
  min_training_bars: 100

hurst:
  trending_threshold: 0.60
  range_min_hurst: 0.48
  range_max_hurst: 0.58

cpd:
  penalty: 5.0
  recency_bars: 5

volatility:
  window: 20
  expanding_threshold: 1.5

range_detection:
  channel_type: "donchian"
  channel_period: 30
  min_bars_persistence: 10

exit_mandate:
  grace_bars: 2
  triggers:
    - "hurst_above_trending"
    - "cpd_structural_break"
    - "volatility_expanding"
    - "hmm_state_change"
```

---

## 10. Validated Integration: zpair (Hyperliquid)

The module has been integrated and validated with live market data in the zpair
pairs trading bot on Hyperliquid.

**Confirmed working output** (BTC, 5m candles, 2026-03-15):

```
 Tick  Time      Price        Consensus          Recommended               Conf  Vol          HMM    Hurst
    1  02:23:54  $70,543.50   CHOP_NEUTRAL       SCALP_MEAN_REVERSION      0.52  MODERATE     CHOP   0.5058
    2  02:24:00  $70,543.50   CHOP_NEUTRAL       RANGE_TRADING             0.51  CONTRACTING  CHOP   0.5058
    3  02:24:05  $70,546.50   CHOP_NEUTRAL       RANGE_TRADING             0.52  CONTRACTING  CHOP   0.5058
```

All signals producing real values from tick 1 (bootstrapped with 300 historical
5m candles). HMM correctly identifies CHOP regime. Recommendation transitions
from SCALP to RANGE as volatility contracts and range persistence builds.

---

## 11. Design Decisions & Constraints

1. **Pure computation** — no I/O, no connectors, no threads. The calling bot controls timing.
2. **Graceful degradation** — any None input produces UNKNOWN for that field only.
3. **HMM warmup** — first 100 bars return UNKNOWN. Bootstrap with historical data to skip.
4. **Exit mandate** — immediate triggers bypass grace period for critical signals.
5. **Fathon optional** — DFA Hurst uses fathon if installed, pure numpy fallback otherwise.
6. **No state persistence** — in-memory only. Bootstrap with historical candles on restart.
7. **Downstream responsibilities** — hold time limits (e.g., scalping 15-min max), position sizing, and specific entry/exit triggers are NOT in this module. This module only provides regime classification and recommendation.
