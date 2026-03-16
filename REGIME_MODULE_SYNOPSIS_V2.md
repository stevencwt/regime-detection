# Regime Detection Module — Integration Guide for AI Systems (v2)

> **Purpose**: This document provides everything an AI assistant needs to install,
> integrate, and validate the `regime-detection` Python package in any trading bot.
> It is designed to be passed directly as context/prompt to an AI system.
>
> **CRITICAL INSTRUCTION TO AI SYSTEM**: After completing integration, you MUST
> create a live test script that proves the regime module is working correctly
> with the bot's real data connectors. See Section 14 for the mandatory test
> script requirements. Do NOT skip this step.

---

## 1. What This Module Does

`regime-detection` is a **pure computation** Python package that classifies market
regime in real-time and recommends which trading playbook to activate. It has:

- **Zero exchange connectors** — no Hyperliquid, no Moomoo, no yfinance, no IBKR
- **Zero I/O** — data flows IN via `.update()`, regime flows OUT via `.get_json()`
- **185 passing tests** across 6 test modules

### The Pipeline

```
Bar data (OHLCV) ──┐
Funding rate ───────┤
Order-book imbalance┤──→ RegimeManager.update()
Options snapshot ───┤         │
Spread data ────────┘         ▼
                        ┌─────────────────┐
                        │ HMM (3-state)   │→ BULL / BEAR / CHOP
                        │ DFA Hurst       │→ 0.0 (mean-revert) ... 1.0 (trend)
                        │ CPD (BinSeg)    │→ structural break True/False
                        │ Volatility      │→ LOW_STABLE / MODERATE / EXPANDING / CONTRACTING
                        │ Liquidity       │→ CONSOLIDATION / TRAP / PASSED
                        │ Consensus vote  │→ BULL_PERSISTENT / BEAR_PERSISTENT / CHOP_NEUTRAL / TRANSITION
                        │ Recommendation  │→ SCALP / RANGE / SWING / PAIRS / OPTIONS / NO_TRADE
                        │ Range hints     │→ Donchian/Keltner boundaries
                        │ Exit mandate    │→ regime shift → force close all
                        └─────────────────┘
                                │
                                ▼
                        RegimeManager.get_json()  →  v3.1 JSON schema
```

---

## 2. Package Location & Installation

### 2.1 Source Repository

```
GitHub: https://github.com/stevencwt/regime-detection
Local:  /Users/user/regime-detection
```

### 2.2 Package Structure

```
regime-detection/
├── pyproject.toml                          # hatchling build system
├── config/
│   └── default_config.yaml                 # all v3.1 thresholds (157 lines)
├── src/regime_detection/
│   ├── __init__.py                         # public API: RegimeManager + enums
│   ├── config.py                           # YAML load → deep merge → validate
│   ├── schema.py                           # enums + dataclasses → v3.1 JSON
│   ├── signals.py                          # HMM, DFA Hurst, CPD, vol, liquidity, funding
│   ├── processors.py                       # crypto/options/pairs context + consensus voting
│   └── recommendation.py                   # recommended logic + range hints + exit mandate
└── tests/
    ├── test_config.py                      # 24 tests
    ├── test_schema.py                      # 13 tests
    ├── test_manager.py                     # 25 tests
    ├── test_signals.py                     # 38 tests
    ├── test_processors.py                  # 42 tests
    └── test_recommendation.py              # 43 tests (185 total)
```

### 2.3 Installation

```bash
# Install as editable dependency into any bot's Python environment
cd /path/to/your/bot
pip install -e /Users/user/regime-detection

# Or with dev dependencies (for running tests)
pip install -e "/Users/user/regime-detection[dev]"

# Verify
python -c "from regime_detection import RegimeManager; print('OK')"
```

### 2.4 Dependencies

Required (installed automatically):
- numpy >= 1.24
- pandas >= 2.0
- pyyaml >= 6.0
- hmmlearn >= 0.3
- ruptures >= 1.1
- scipy >= 1.11

Optional:
- fathon >= 1.3 (faster DFA Hurst; pure numpy fallback used if absent)

---

## 3. Public API Reference

### 3.1 Constructor

```python
from regime_detection import RegimeManager

manager = RegimeManager(
    config_path=None,          # str | Path — override YAML, or None for defaults
    market_type="CRYPTO_PERP", # "CRYPTO_PERP" | "CRYPTO_SPOT" | "US_STOCK" | "US_OPTION" | "SPREAD"
    strategy_type="scalping",  # "scalping" | "range_trading" | "swing" | "pairs_trading" | "options_income" | "options_speculative"
    market_class="crypto",     # "crypto" | "us_stocks"
)
```

### 3.2 .update() — Feed Data

```python
manager.update(
    bar={                              # REQUIRED — OHLCV dict
        "timestamp": 1710000000,       #   epoch seconds or ISO string
        "o": 84200.0,                  #   open
        "h": 84350.0,                  #   high
        "l": 84100.0,                  #   low
        "c": 84250.0,                  #   close
        "v": 42000.0,                  #   volume
    },
    funding_rate=0.0003,               # OPTIONAL — crypto perpetual funding rate (float)
    order_book_imbalance=-0.12,        # OPTIONAL — normalized to [-1, +1] (float)
    options_snapshot={                  # OPTIONAL — US stocks only
        "vanna": 0.45,                 #   net vanna signal
        "gamma": -0.3,                 #   net gamma exposure (or "gamma_flip": 4500.0)
        "oi_skew": 1.2,               #   call/put OI ratio
    },
    spread_data={                      # OPTIONAL — pairs trading only
        "spread_series": [0.1, -0.3, 0.5, ...],  # list[float] — historical spread values
        "half_life": 18.0,             #   spread half-life in bars
        "cointegration_pvalue": 0.03,  #   ADF p-value
    },
)
```

**Key rules:**
- `bar` is the only required argument. All others are keyword-only and optional.
- Minimum required bar keys: `o`, `h`, `l`, `c`, `v` (all numeric).
- Missing optional data → that field shows "UNKNOWN" in output, others still computed.
- First ~100 bars are warmup (HMM training). Output stays UNKNOWN/NO_TRADE until then.
- **To skip warmup**: bootstrap with historical candles before starting the live loop.

### 3.3 .get_current_regime() — Read Output as Dict

```python
regime = manager.get_current_regime()  # returns dict
```

### 3.4 .get_json() — Read Output as JSON String

```python
json_str = manager.get_json(indent=2)  # returns formatted JSON string
```

### 3.5 .reload_config() — Hot-Reload Config

```python
issues = manager.reload_config()  # re-reads YAML, returns list of warnings
```

### 3.6 .bar_count — Bars in Buffer

```python
n = manager.bar_count  # int — how many bars currently buffered
```

---

## 4. Output Schema (v3.1 JSON)

Every call to `.get_json()` produces this exact structure:

```json
{
  "consensus_state": "CHOP_NEUTRAL",
  "market_type": "CRYPTO_PERP",
  "confidence_score": 0.72,
  "volatility_regime": "LOW_STABLE",
  "signals": {
    "hmm_label": "CHOP",
    "hurst_dfa": 0.4832,
    "structural_break": false,
    "liquidity_status": "CONSOLIDATION",
    "crypto_context": {
      "funding_bias": "NEUTRAL",
      "funding_rate": 0.0003,
      "order_book_imbalance": -0.12
    },
    "range_hints": {
      "is_clean_range": true,
      "range_lower": 84100.0,
      "range_upper": 84400.0,
      "current_deviation_pct": -1.4,
      "channel_type": "Donchian_30"
    }
  },
  "recommended_logic": "SCALP_MEAN_REVERSION",
  "exit_mandate": false,
  "timestamp": "2026-03-14T14:30:00+00:00"
}
```

### 4.1 Field Reference

| Field | Type | Values |
|---|---|---|
| `consensus_state` | string | `BULL_PERSISTENT`, `BEAR_PERSISTENT`, `CHOP_NEUTRAL`, `TRANSITION`, `UNKNOWN` |
| `market_type` | string | `CRYPTO_PERP`, `CRYPTO_SPOT`, `US_STOCK`, `US_OPTION`, `SPREAD` |
| `confidence_score` | float | 0.0 – 1.0 |
| `volatility_regime` | string | `LOW_STABLE`, `MODERATE`, `EXPANDING`, `CONTRACTING`, `UNKNOWN` |
| `recommended_logic` | string | `SCALP_MEAN_REVERSION`, `RANGE_TRADING`, `SWING_TREND_FOLLOW`, `PAIRS_MEAN_REVERSION`, `OPTIONS_INCOME`, `OPTIONS_SPECULATIVE`, `NO_TRADE` |
| `exit_mandate` | bool | `true` = regime shifted → close all positions |
| `signals.hmm_label` | string | `BULL`, `BEAR`, `CHOP`, `UNKNOWN` |
| `signals.hurst_dfa` | float\|null | 0.0 (mean-revert) → 0.5 (random) → 1.0 (trending) |
| `signals.structural_break` | bool | CPD detected a recent change point |
| `signals.liquidity_status` | string | `CONSOLIDATION`, `LIQUIDITY_TRAP`, `PASSED`, `UNKNOWN` |

### 4.2 Optional Contexts (appear only when relevant)

- `signals.crypto_context` — present when `market_type` is CRYPTO_PERP or CRYPTO_SPOT
- `signals.options_context` — present when `market_type` is US_STOCK or US_OPTION
- `signals.pairs_context` — present when `market_type` is SPREAD
- `signals.range_hints` — present when consensus is CHOP_NEUTRAL and logic is SCALP or RANGE

---

## 5. Recommended Logic Activation Rules

These are the v3.1 rules that determine `recommended_logic`:

| Condition | Result |
|---|---|
| UNKNOWN / TRANSITION / structural break / None Hurst | `NO_TRADE` |
| `strategy_type="pairs_trading"` + CHOP_NEUTRAL | `PAIRS_MEAN_REVERSION` |
| `strategy_type="pairs_trading"` + trending | `NO_TRADE` |
| `strategy_type="options_income"` + stable vol | `OPTIONS_INCOME` |
| `strategy_type="options_income"` + EXPANDING vol | `NO_TRADE` |
| `strategy_type="options_speculative"` + trending or expanding | `OPTIONS_SPECULATIVE` |
| BULL_PERSISTENT or BEAR_PERSISTENT | `SWING_TREND_FOLLOW` |
| CHOP_NEUTRAL + Hurst < 0.48 + CONSOLIDATION | `SCALP_MEAN_REVERSION` |
| CHOP_NEUTRAL + Hurst 0.48–0.58 + LOW_STABLE vol + persistence ≥ 10 bars | `RANGE_TRADING` |
| CHOP_NEUTRAL + Hurst 0.48–0.58 + persistence not met | `SCALP_MEAN_REVERSION` (interim) |
| CHOP_NEUTRAL + LIQUIDITY_TRAP | `NO_TRADE` |

### Exit Mandate Triggers

| Trigger | Grace Period? |
|---|---|
| CPD structural break | Immediate |
| Hurst ≥ 0.60 (was CHOP) | Immediate |
| Volatility → EXPANDING (was CHOP) | Immediate |
| Consensus state change (e.g. CHOP→BULL) | 2-bar confirmation |

---

## 6. Temporal Matrix (Strategy ↔ Timeframe Mapping)

The `strategy_type` and `market_class` determine lookback window and HMM stability:

| Strategy | Market | Signal TF | Execution TF | Lookback | HMM Stability |
|---|---|---|---|---|---|
| scalping | crypto | 5m | 1m | 750 bars | 2 bars |
| scalping | us_stocks | 5m | 1m | 500 bars | 3 bars |
| range_trading | crypto | 15m | 5m | 1000 bars | 3 bars |
| range_trading | us_stocks | 1h | 15m | 800 bars | 4 bars |
| swing | crypto | 1h | 15m | 1000 bars | 3 bars |
| swing | us_stocks | 1h | 15m | 800 bars | 4 bars |
| options_income | us_stocks | 1d | 1d | 252 bars | 5 bars |
| options_speculative | us_stocks | 1d | 30m | 252 bars | 4 bars |
| pairs_trading | crypto | 15m | 5m | 1000 bars | 3 bars |

---

## 7. HMM Robustness Features (Important)

The HMM module includes three robustness guards that integrators should understand.
These were added after real-world testing with live Hyperliquid BBO data, where
5-second polling produced near-identical consecutive prices that caused the HMM
to oscillate between states on every bar.

### 7.1 Near-Zero Variance Guard

When consecutive prices are identical (e.g., polling BBO of an illiquid asset,
or during exchange maintenance), log-returns have near-zero variance. The HMM
cannot differentiate states from noise. The guard detects `std(log_returns) < 1e-10`
and returns `CHOP` with 0.5 confidence instead of attempting a meaningless fit.

**Implication for integrators**: If you poll live prices faster than the candle
timeframe, many consecutive bars will have identical close prices. This is handled
gracefully — you will get `CHOP` rather than `UNKNOWN`. For better HMM quality,
bootstrap with proper OHLCV candles at the signal timeframe (e.g., 5m candles).

### 7.2 Direction-Aware Means Fallback

When the HMM fits 3 states but all state means cluster together (e.g., pure
uptrend where every return is positive), the states are meaningless but the
overall direction is clear. Instead of defaulting to CHOP, the module checks
the average of all state means:
- Average strongly positive → `BULL`
- Average strongly negative → `BEAR`
- Average near zero → `CHOP`

### 7.3 Majority Vote Stability

The stability check uses a **majority vote** over the last 5 bars (or the
configured `hmm_stability_bars`, whichever is larger). The most-common label
needs >50% to win. If no label has a majority, it defaults to `CHOP` (market
is transitioning). This replaces the earlier unanimity check which caused
persistent `UNKNOWN` output during noisy periods.

---

## 8. Integration Pattern (Step-by-Step for AI Systems)

Follow this exact pattern when integrating into any trading bot:

### Step 1: Identify the bot's data source

Find where the bot obtains OHLCV bars. Common patterns:
- WebSocket streaming (on_message callback with candle data)
- REST polling (fetch candles every N seconds)
- Existing data pipeline (telemetry service, buffer system, etc.)

### Step 2: Identify available optional data

Check if the bot has access to:
- **Funding rate** (crypto perps) — from exchange API metadata
- **Order-book imbalance** — from L2 book data or BBO bid/ask sizes
- **Options data** (US stocks) — from options chain API
- **Spread data** (pairs) — from computed spread series

### Step 3: Create a bridge module

Create a single file (e.g., `regime_bridge.py`) in the bot's directory that:
1. Imports `RegimeManager` from the package
2. Imports the bot's existing data connector/adapter
3. Provides one function that extracts data from the bot's format and calls `.update()`

### Step 4: Wire into the bot's main loop

Add 3 blocks to the bot's main file:
1. **Import** the bridge (with try/except fallback)
2. **Initialize** in the setup/init section
3. **Call** the bridge's update method in the tick/loop body

### Step 5: Create a standalone live test script (MANDATORY — see Section 14)

---

## 9. Integration Template Code

### 9.1 Generic Bridge Module Template

```python
"""
regime_bridge.py — Drop into your bot's directory.
Adapt _extract_bar() and _get_* methods for your data source.
"""
import time
import math
from typing import Any, Dict, Optional

try:
    from regime_detection import RegimeManager
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False
    print("[REGIME] Install: pip install -e /Users/user/regime-detection")


class RegimeBridge:
    def __init__(self, market_type="CRYPTO_PERP", strategy_type="scalping",
                 market_class="crypto", config_path=None):
        self.enabled = REGIME_AVAILABLE
        if not self.enabled:
            return

        self.manager = RegimeManager(
            config_path=config_path,
            market_type=market_type,
            strategy_type=strategy_type,
            market_class=market_class,
        )
        self._tick = 0

    def update(self, bar: dict, funding_rate=None,
               order_book_imbalance=None, options_snapshot=None,
               spread_data=None) -> dict:
        """Feed one bar, return regime dict."""
        if not self.enabled:
            return {}

        self.manager.update(
            bar,
            funding_rate=funding_rate,
            order_book_imbalance=order_book_imbalance,
            options_snapshot=options_snapshot,
            spread_data=spread_data,
        )
        self._tick += 1

        # Log every 20 ticks
        if self._tick % 20 == 0:
            r = self.manager.get_current_regime()
            print(f"[REGIME] {r['consensus_state']} | {r['recommended_logic']} | "
                  f"conf={r['confidence_score']:.2f} hmm={r['signals']['hmm_label']} "
                  f"hurst={r['signals']['hurst_dfa']} mandate={r['exit_mandate']}")

        return self.manager.get_current_regime()

    def get_regime(self) -> dict:
        if not self.enabled:
            return {}
        return self.manager.get_current_regime()

    def is_exit_mandated(self) -> bool:
        if not self.enabled:
            return False
        return self.manager.get_current_regime().get("exit_mandate", False)
```

### 9.2 Main Loop Integration Template

```python
# --- In your bot's main file ---

# 1. IMPORT (at the top)
try:
    from regime_bridge import RegimeBridge
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False

# 2. INITIALIZE (in setup/init)
regime_bridge = None
if REGIME_AVAILABLE:
    try:
        regime_bridge = RegimeBridge(
            market_type="CRYPTO_PERP",     # adapt for your market
            strategy_type="scalping",       # adapt for your strategy
            market_class="crypto",          # "crypto" or "us_stocks"
        )
    except Exception as e:
        print(f"Regime init failed: {e}")

# 3. UPDATE (in your main loop, after you have bar data)
if regime_bridge:
    bar = {
        "timestamp": time.time(),
        "o": candle["open"],      # adapt field names for your data source
        "h": candle["high"],
        "l": candle["low"],
        "c": candle["close"],
        "v": candle["volume"],
    }
    regime = regime_bridge.update(
        bar,
        funding_rate=your_funding_rate,        # or None
        order_book_imbalance=your_imbalance,   # or None
    )

    if regime.get("exit_mandate"):
        # CLOSE ALL POSITIONS — regime shifted
        close_all_positions()
```

---

## 10. Existing Integration: zpair (Hyperliquid Pairs Trading)

This is the reference integration already completed and validated with live data:

### Bot: zpair
- **Location**: `/Users/user/zpair`
- **Repository**: `https://github.com/stevencwt/zpair`
- **Connector**: `hyperliquid_utils_adapter.py` → `HyperliquidAdapter` class
- **Data methods**:
  - `adapter.get_price(coin)` → float (current mid price)
  - `adapter.get_candles(coin, interval, bars)` → list of `{"t","o","h","l","c","v"}`
  - `adapter.get_bbo(coin)` → `{"bid", "ask", "mid"}`
  - `adapter.info.meta_and_asset_ctxs()` → funding rates (index-matched to universe)

### Files added to zpair:
- `regime_bridge.py` — extracts tele_wrapped data → feeds RegimeManager
- `vmain_v14s.py` — 3 additive blocks (import, init, update in loop)
- `test_regime_live.py` — standalone live test using HyperliquidAdapter

### Live validation output (confirmed working):
```
    1  02:23:54    $70,543.50      0.1b  CHOP_NEUTRAL        SCALP_MEAN_REVERSION       0.52  MODERATE      CHOP     0.5058   n   CONSOLIDATION      no
    2  02:24:00    $70,543.50      0.1b  CHOP_NEUTRAL        RANGE_TRADING              0.51  CONTRACTING   CHOP     0.5058   n   CONSOLIDATION      no
    3  02:24:05    $70,546.50      0.1b  CHOP_NEUTRAL        RANGE_TRADING              0.52  CONTRACTING   CHOP     0.5058   n   CONSOLIDATION      no
```

All signals producing real values from tick 1 (bootstrapped with 300 historical 5m candles).

---

## 11. Config Override Examples

Create a YAML file to override any defaults:

```yaml
# regime_override.yaml — place in your bot's directory

# Tighter Hurst thresholds
hurst:
  range_min_hurst: 0.46
  range_max_hurst: 0.56
  trending_threshold: 0.62

# More sensitive break detection
cpd:
  penalty: 3.0
  recency_bars: 3

# Faster exit mandate
exit_mandate:
  grace_bars: 1

# Longer HMM warmup for noisy data
hmm:
  min_training_bars: 150
  n_iter: 80
```

Pass to constructor:
```python
manager = RegimeManager(config_path="regime_override.yaml", ...)
```

---

## 12. Key Design Decisions

1. **Pure computation** — no I/O, no connectors, no threads. The calling bot controls timing.
2. **Graceful degradation** — any None input produces UNKNOWN for that field only. Other fields still compute.
3. **HMM warmup** — first 100 bars return UNKNOWN. Bootstrap with historical data to avoid this.
4. **HMM robustness** — near-zero variance guard, direction-aware fallback, majority vote stability (see Section 7).
5. **Exit mandate** — immediate on structural break, Hurst spike, or vol expansion. Grace period (2 bars default) for consensus state changes.
6. **Fathon optional** — DFA Hurst uses fathon if installed, falls back to pure numpy implementation.
7. **No state persistence** — RegimeManager is in-memory only. If the bot restarts, bootstrap with historical candles again.

---

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| HMM=CHOP with low confidence (~0.5) | Near-zero variance data or undifferentiated states | Expected for flat/ranging markets or stale data. Bootstrap with proper candles. |
| All UNKNOWN after 100+ bars | HMM not reaching stability (rare after v2 fix) | Check if price data has actual variation. Constant prices will produce CHOP, not UNKNOWN. |
| consensus_state always CHOP_NEUTRAL | Hurst < 0.60 (below trending threshold) | Expected for ranging markets. This is correct behavior. |
| recommended_logic always NO_TRADE | Safety gate triggered (UNKNOWN/TRANSITION/break) | Check signals — likely structural_break is True or consensus is TRANSITION. |
| recommended_logic is SCALP then flips to RANGE | Hurst entered 0.48-0.58 sweet spot + vol contracting + persistence building | Normal — the module correctly distinguishes scalp vs range conditions. |
| ImportError on `from regime_detection` | Package not installed in this Python env | Run `pip install -e /Users/user/regime-detection` |
| fathon install fails on macOS | No pre-built wheel for macOS/arm64 | Safe to ignore — pure numpy DFA fallback is used automatically. |
| HMM convergence warnings | Normal for small datasets | Suppressed internally. Does not affect output. |

---

## 14. MANDATORY: Live Test Script Creation

> **INSTRUCTION TO AI SYSTEM**: After installing the regime-detection package
> and integrating it into the bot, you MUST create a live test script that
> validates the module is working correctly with the bot's real data.
> Do NOT tell the user "it should work" — prove it works by creating and
> running this test script.

### 14.1 Requirements for the Test Script

The test script MUST:

1. **Be named** `test_regime_live.py` and placed in the bot's root directory.

2. **Use the bot's existing data connector** — do NOT create new API clients.
   Import the same adapter/connector class the bot already uses to fetch market data.

3. **Accept command-line arguments**:
   - `--asset` (default: the bot's primary asset, e.g., "BTC", "AAPL", "SPY")
   - `--timeframe` (default: "5m")
   - `--poll` (default: 5 seconds)
   - `--history` (default: 300 bars)
   - `--strategy` (default: appropriate for the bot, e.g., "scalping", "swing", "options_income")

4. **Print a startup banner** showing:
   - Asset name
   - Timeframe
   - Strategy type
   - Poll interval
   - Number of historical bars to bootstrap
   - The connector being used (e.g., "HyperliquidAdapter", "MoomooClient", "yfinance")

5. **Bootstrap with historical candles** — fetch 200-300 historical OHLCV candles
   using the bot's existing connector and feed them to RegimeManager BEFORE
   starting the live loop. This skips the HMM warmup period.
   - Print: how many bars were fed, price range of historical data
   - Print: initial regime state after bootstrap

6. **Run a continuous live loop** (exit with Ctrl+C) that on each tick:
   - Fetches current price/BBO from the existing connector
   - Optionally fetches funding rate (crypto) or options data (stocks) if available
   - Optionally computes order-book imbalance if L2 data is available
   - Constructs a bar dict: `{"timestamp", "o", "h", "l", "c", "v"}`
   - Feeds it to `RegimeManager.update()`
   - Prints a one-line summary with ALL of these columns:

   ```
   Tick  Time      Price        Spread   Consensus          Recommended               Conf   Vol          HMM      Hurst   Brk  Liq             Exit
     1   14:30:05  $84,231.50   1.2b     CHOP_NEUTRAL       SCALP_MEAN_REVERSION      0.68   LOW_STABLE   CHOP    0.4832    n   CONSOLIDATION    no
   ```

7. **Print a detailed regime box every 20 ticks** showing:
   ```
   ┌─ BTC 5m Regime Detail (tick 20)
   │  Consensus:    CHOP_NEUTRAL (conf=0.710)
   │  Recommended:  SCALP_MEAN_REVERSION
   │  HMM state:    CHOP
   │  Hurst DFA:    0.4912
   │  Volatility:   MODERATE
   │  Break detect: False
   │  Liquidity:    CONSOLIDATION
   │  Funding:      0.000100 (NEUTRAL)
   │  Imbalance:    -0.120
   │  Range hints:  [$84,100 — $84,400] dev=-2.1% clean=no (Donchian_30)
   │  Exit mandate: False
   │  Bars in buffer: 320
   └─────────────────────────────
   ```

8. **On Ctrl+C**, print:
   - Total ticks, total bars in buffer
   - Final full JSON output from `manager.get_json()`

### 14.2 Validation Criteria

After running the test script for at least 20 ticks, confirm:

| Check | Expected | If Wrong |
|---|---|---|
| HMM label | NOT "UNKNOWN" (should be BULL, BEAR, or CHOP) | Bootstrap didn't work or candle data is invalid |
| Hurst DFA | A float between 0.0 and 1.0 (not None) | Close prices may all be identical or insufficient |
| Volatility | NOT "UNKNOWN" (should be LOW_STABLE, MODERATE, etc.) | Need more price history |
| Consensus | NOT "UNKNOWN" (should be CHOP_NEUTRAL, BULL_PERSISTENT, etc.) | HMM or Hurst is failing |
| Recommended logic | Should be a real value (SCALP, RANGE, SWING, etc.) or NO_TRADE with clear reason | Check consensus — if UNKNOWN, recommendation will be NO_TRADE |
| Liquidity | NOT "UNKNOWN" if order_book_imbalance is being passed | Check imbalance value is in [-1, +1] |
| Funding (crypto) | "NEUTRAL", "EXTREME_POSITIVE", or "EXTREME_NEGATIVE" if funding_rate passed | Check funding rate value |

**If ALL signals show real values (not UNKNOWN/None) from tick 1**, the integration
is validated. Report this to the user.

### 14.3 Common Pitfall: BBO Polling vs Candle Timeframe

**WARNING**: If the test script polls BBO every 5 seconds, the bar buffer will
eventually fill with near-identical close prices. The HMM handles this gracefully
(returns CHOP via the near-zero variance guard) but the Hurst and other signals
may become less informative. This is acceptable for a validation test.

For production quality, the bridge module should feed proper OHLCV candles at the
signal timeframe (e.g., one 5m candle every 5 minutes) rather than BBO ticks.

### 14.4 Template Structure

```python
#!/usr/bin/env python3
"""test_regime_live.py — Live regime test using [BOT_NAME] connector."""

import argparse, time, sys
from datetime import datetime

# ==============================================================
# ADAPT THESE IMPORTS for your bot's connector
# ==============================================================
from your_bot.connector import YourAdapter  # <-- CHANGE THIS
# ==============================================================

from regime_detection import RegimeManager


def load_connector():
    """Initialize your bot's data connector."""
    # <-- ADAPT: use your bot's connection/auth method
    pass


def fetch_historical_candles(connector, asset, timeframe, count):
    """Fetch historical OHLCV candles.
    Returns list of dicts: [{"timestamp", "o", "h", "l", "c", "v"}, ...]
    """
    # <-- ADAPT: call your connector's candle method
    # <-- ADAPT: normalize field names to o/h/l/c/v
    pass


def fetch_live_price(connector, asset):
    """Fetch current price (and optionally BBO).
    Returns (price, bid, ask) or (price, None, None)
    """
    # <-- ADAPT: call your connector's price/BBO method
    pass


def fetch_funding_rate(connector, asset):
    """Fetch funding rate. Return float or None."""
    # <-- ADAPT: only for crypto perps, return None for stocks
    pass


def fetch_order_book_imbalance(connector, asset):
    """Compute order-book imbalance from L2 data. Return float [-1,+1] or None."""
    # <-- ADAPT: compute from your connector's order book data
    pass


def fetch_options_snapshot(connector, asset):
    """Fetch options data. Return dict or None."""
    # <-- ADAPT: only for US stocks, return None for crypto
    pass


def main():
    parser = argparse.ArgumentParser(description="Live Regime Detection Test")
    parser.add_argument("--asset", default="BTC")           # <-- ADAPT default
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--poll", type=int, default=5)
    parser.add_argument("--history", type=int, default=300)
    parser.add_argument("--strategy", default="scalping")    # <-- ADAPT default
    args = parser.parse_args()

    # --- Banner ---
    print(f"\n{'='*70}")
    print(f"  LIVE REGIME DETECTION — [BOT_NAME] Real Data")  # <-- ADAPT name
    print(f"{'='*70}")
    print(f"  Asset:        {args.asset}")
    print(f"  Timeframe:    {args.timeframe}")
    print(f"  Strategy:     {args.strategy}")
    print(f"  Poll interval: {args.poll}s")
    print(f"  History bars: {args.history}")
    print(f"  Connector:    [YourAdapter]")                    # <-- ADAPT name
    print(f"{'='*70}\n")

    # --- Init connector ---
    connector = load_connector()

    # --- Init RegimeManager ---
    manager = RegimeManager(
        market_type="CRYPTO_PERP",       # <-- ADAPT: US_STOCK for stocks
        strategy_type=args.strategy,
        market_class="crypto",            # <-- ADAPT: "us_stocks" for stocks
    )
    min_bars = manager.config["hmm"]["min_training_bars"]
    print(f"[INIT] HMM warmup threshold: {min_bars} bars")

    # --- Bootstrap ---
    print(f"[BOOTSTRAP] Fetching {args.history} historical {args.timeframe} candles...")
    candles = fetch_historical_candles(connector, args.asset, args.timeframe, args.history)
    for c in candles:
        manager.update(c)
    print(f"[BOOTSTRAP] Fed {len(candles)} bars")
    regime = manager.get_current_regime()
    print(f"[BOOTSTRAP] Initial: {regime['consensus_state']} | {regime['recommended_logic']}\n")

    # --- Live loop ---
    print(f"[LIVE] Polling every {args.poll}s. Ctrl+C to stop.")
    print(f"{'─'*100}")
    print(f"{'Tick':>5s}  {'Time':>8s}  {'Price':>12s}  "
          f"{'Consensus':<18s}  {'Recommended':<24s}  "
          f"{'Conf':>5s}  {'Vol':<12s}  {'HMM':<7s}  {'Hurst':>7s}  "
          f"{'Brk':>3s}  {'Liq':<15s}  {'Exit':>4s}")
    print(f"{'─'*100}")

    tick = 0
    prev_price = float(candles[-1]["c"]) if candles else 0

    try:
        while True:
            tick += 1
            price, bid, ask = fetch_live_price(connector, args.asset)
            if price is None or price <= 0:
                time.sleep(args.poll)
                continue

            bar = {
                "timestamp": time.time(),
                "o": prev_price,
                "h": max(price, prev_price, ask or price),
                "l": min(price, prev_price, bid or price),
                "c": price,
                "v": 0.0,
            }

            funding = fetch_funding_rate(connector, args.asset) if tick % 12 == 1 else None
            imbalance = fetch_order_book_imbalance(connector, args.asset)
            options = fetch_options_snapshot(connector, args.asset)

            manager.update(bar, funding_rate=funding,
                          order_book_imbalance=imbalance,
                          options_snapshot=options)

            regime = manager.get_current_regime()
            sig = regime["signals"]
            hurst = sig.get("hurst_dfa")
            hurst_str = f"{hurst:.4f}" if hurst is not None else "  N/A "

            print(
                f"{tick:5d}  {datetime.now():%H:%M:%S}  "
                f"${price:>10,.2f}  "
                f"{regime['consensus_state']:<18s}  "
                f"{regime['recommended_logic']:<24s}  "
                f"{regime['confidence_score']:5.2f}  "
                f"{regime['volatility_regime']:<12s}  "
                f"{sig['hmm_label']:<7s}  {hurst_str}  "
                f"{'Y' if sig['structural_break'] else 'n':>3s}  "
                f"{sig['liquidity_status']:<15s}  "
                f"{'YES' if regime['exit_mandate'] else 'no':>4s}"
            )

            # Detailed box every 20 ticks
            if tick % 20 == 0:
                crypto = sig.get("crypto_context", {})
                rh = sig.get("range_hints")
                fr = crypto.get("funding_rate")
                print(f"{'─'*100}")
                print(f"  ┌─ {args.asset} {args.timeframe} Regime Detail (tick {tick})")
                print(f"  │  Consensus:    {regime['consensus_state']} (conf={regime['confidence_score']:.3f})")
                print(f"  │  Recommended:  {regime['recommended_logic']}")
                print(f"  │  HMM state:    {sig['hmm_label']}")
                print(f"  │  Hurst DFA:    {hurst_str.strip()}")
                print(f"  │  Volatility:   {regime['volatility_regime']}")
                print(f"  │  Break detect: {sig['structural_break']}")
                print(f"  │  Liquidity:    {sig['liquidity_status']}")
                if fr is not None:
                    print(f"  │  Funding:      {fr:.6f} ({crypto.get('funding_bias','?')})")
                if rh:
                    print(f"  │  Range:        [{rh['range_lower']:.2f} — {rh['range_upper']:.2f}] "
                          f"dev={rh['current_deviation_pct']:.1f}% ({rh['channel_type']})")
                print(f"  │  Exit mandate: {regime['exit_mandate']}")
                print(f"  │  Bars:         {manager.bar_count}")
                print(f"  └─────────────────────────────")
                print(f"{'─'*100}")

            prev_price = price
            time.sleep(args.poll)

    except KeyboardInterrupt:
        print(f"\n{'='*70}")
        print(f"  STOPPED — {tick} ticks, {manager.bar_count} bars in buffer")
        print(f"{'='*70}")
        print(f"\nFinal regime JSON:")
        print(manager.get_json())


if __name__ == "__main__":
    main()
```

**Lines marked `# <-- ADAPT` or `# <-- CHANGE THIS` must be customized for
each bot's specific data connector.** Everything else should work as-is.

### 14.5 What "Success" Looks Like

After running for 20+ ticks, the output should look similar to this
(values will differ based on market conditions):

```
  Asset:        BTC
  Timeframe:    5m
  Strategy:     scalping
  Connector:    HyperliquidAdapter
[BOOTSTRAP] Fed 301 bars
[BOOTSTRAP] Initial: CHOP_NEUTRAL | SCALP_MEAN_REVERSION

 Tick      Time         Price  Consensus           Recommended               Conf  Vol           HMM      Hurst  Brk  Liq              Exit
    1  14:30:05    $70,543.50  CHOP_NEUTRAL        SCALP_MEAN_REVERSION      0.52  MODERATE      CHOP    0.5058   n   CONSOLIDATION      no
    2  14:30:10    $70,543.50  CHOP_NEUTRAL        RANGE_TRADING             0.51  CONTRACTING   CHOP    0.5058   n   CONSOLIDATION      no
    3  14:30:15    $70,546.50  CHOP_NEUTRAL        RANGE_TRADING             0.52  CONTRACTING   CHOP    0.5058   n   CONSOLIDATION      no
```

**Validation checklist** (report these to the user):
- [ ] HMM is NOT "UNKNOWN" → ✅ or ❌
- [ ] Hurst is a float (not None) → ✅ or ❌
- [ ] Volatility is NOT "UNKNOWN" → ✅ or ❌
- [ ] Consensus is NOT "UNKNOWN" → ✅ or ❌
- [ ] Recommended logic is a real value → ✅ or ❌
- [ ] No Python errors or crashes → ✅ or ❌

If all checks pass, the regime-detection module is working correctly for this bot.
