# Regime Detection Module — Integration Guide for AI Systems

> **Purpose**: This document provides everything an AI assistant needs to install,
> integrate, and test the `regime-detection` Python package in any trading bot.
> It is designed to be passed directly as context/prompt to an AI system.

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

## 7. Integration Pattern (Step-by-Step for AI Systems)

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

### Step 5: Create a standalone test script

Create a test script that:
1. Uses the bot's existing connector to fetch real data
2. Bootstraps RegimeManager with historical candles
3. Polls live prices in a loop
4. Prints regime state to console

---

## 8. Integration Template Code

### 8.1 Generic Bridge Module Template

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

### 8.2 Main Loop Integration Template

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

## 9. Live Test Script Template

### Instructions for the AI system:

**Create a test script** named `test_regime_live.py` in the bot's directory.
The script must:

1. **Use the bot's existing data connector** (NOT create a new one) to:
   - Fetch 200–300 historical OHLCV candles for a single asset
   - Poll live prices every 3–5 seconds

2. **Bootstrap** RegimeManager with the historical candles (this skips the HMM warmup)

3. **Run a continuous loop** that:
   - Fetches current price / BBO from the existing connector
   - Constructs a bar dict and feeds it to RegimeManager
   - Prints a one-line regime summary every tick
   - Prints a detailed regime box every 20 ticks
   - Exits cleanly on Ctrl+C with final JSON dump

4. **Accept command-line arguments** for:
   - `--asset` (default: the bot's primary asset)
   - `--timeframe` (default: "5m")
   - `--poll` (default: 5 seconds)
   - `--strategy` (default: depends on bot type)

### Reference implementation structure:

```python
#!/usr/bin/env python3
"""test_regime_live.py — Live regime test using existing bot connector."""

import argparse
import time
import sys
from datetime import datetime

# Import YOUR bot's existing connector
from your_bot.connector import YourDataAdapter  # <-- ADAPT THIS

# Import regime detection
from regime_detection import RegimeManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--poll", type=int, default=5)
    parser.add_argument("--history", type=int, default=300)
    parser.add_argument("--strategy", default="scalping")
    args = parser.parse_args()

    # --- Connect using existing adapter ---
    adapter = YourDataAdapter(...)  # <-- ADAPT: use your bot's connection method

    # --- Init RegimeManager ---
    manager = RegimeManager(
        market_type="CRYPTO_PERP",      # <-- ADAPT for your market
        strategy_type=args.strategy,
        market_class="crypto",           # <-- ADAPT: "crypto" or "us_stocks"
    )

    # --- Banner ---
    print(f"Asset: {args.asset} | TF: {args.timeframe} | Strategy: {args.strategy}")
    print(f"HMM warmup: {manager.config['hmm']['min_training_bars']} bars")

    # --- Bootstrap with historical candles ---
    candles = adapter.get_candles(args.asset, args.timeframe, args.history)
    for c in candles:
        bar = {
            "timestamp": c["t"],   # <-- ADAPT field names
            "o": float(c["o"]),
            "h": float(c["h"]),
            "l": float(c["l"]),
            "c": float(c["c"]),
            "v": float(c["v"]),
        }
        manager.update(bar)
    print(f"Bootstrapped with {len(candles)} bars")

    # --- Live loop ---
    tick = 0
    try:
        while True:
            tick += 1
            price = adapter.get_price(args.asset)  # <-- ADAPT

            bar = {
                "timestamp": time.time(),
                "o": price, "h": price, "l": price, "c": price,
                "v": 0.0,
            }

            funding = None       # <-- ADAPT: get from your API if available
            imbalance = None     # <-- ADAPT: get from your order book if available

            manager.update(bar, funding_rate=funding, order_book_imbalance=imbalance)

            regime = manager.get_current_regime()
            sig = regime["signals"]

            hurst = sig.get("hurst_dfa")
            hurst_str = f"{hurst:.4f}" if hurst else "N/A"

            print(
                f"[{tick:4d}] {datetime.now():%H:%M:%S} "
                f"price={price:.2f} "
                f"{regime['consensus_state']:<18s} | "
                f"{regime['recommended_logic']:<24s} | "
                f"hmm={sig['hmm_label']:<7s} hurst={hurst_str} "
                f"vol={regime['volatility_regime']:<12s} "
                f"mandate={regime['exit_mandate']}"
            )

            # Detailed box every 20 ticks
            if tick % 20 == 0:
                print(f"\n{'─'*60}")
                print(f"  {args.asset} {args.timeframe} Detail (tick {tick})")
                print(f"  Consensus:    {regime['consensus_state']}")
                print(f"  Recommended:  {regime['recommended_logic']}")
                print(f"  Confidence:   {regime['confidence_score']:.3f}")
                print(f"  HMM:          {sig['hmm_label']}")
                print(f"  Hurst:        {hurst_str}")
                print(f"  Volatility:   {regime['volatility_regime']}")
                print(f"  Break:        {sig['structural_break']}")
                print(f"  Liquidity:    {sig['liquidity_status']}")
                print(f"  Mandate:      {regime['exit_mandate']}")
                print(f"  Bars:         {manager.bar_count}")
                print(f"{'─'*60}\n")

            time.sleep(args.poll)

    except KeyboardInterrupt:
        print(f"\nStopped after {tick} ticks")
        print(manager.get_json())


if __name__ == "__main__":
    main()
```

**Lines marked `# <-- ADAPT` must be changed for each bot's specific connector.**

---

## 10. Existing Integration: zpair (Hyperliquid Pairs Trading)

This is the reference integration already completed:

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

### zpair integration details:
- `strategy_type="pairs_trading"` → gets `PAIRS_MEAN_REVERSION` in CHOP
- `market_type="CRYPTO_PERP"` → enables `crypto_context` (funding bias)
- Bootstrap: 300 historical 5m candles from `adapter.get_candles()`
- Live: BBO polling every 5s, funding rate every 60s

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
4. **Exit mandate** — immediate on structural break, Hurst spike, or vol expansion. Grace period (2 bars default) for consensus state changes.
5. **Fathon optional** — DFA Hurst uses fathon if installed, falls back to pure numpy implementation.
6. **No state persistence** — RegimeManager is in-memory only. If the bot restarts, bootstrap with historical candles again.

---

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| All UNKNOWN after 100+ bars | HMM stability check failing — label oscillating | Normal for low-volatility periods. Wait for more varied price action. |
| consensus_state always CHOP_NEUTRAL | Hurst < 0.60 (below trending threshold) | Expected for ranging markets. Check if Hurst value makes sense. |
| recommended_logic always NO_TRADE | Safety gate triggered (UNKNOWN/TRANSITION/break) | Check signals — likely HMM is UNKNOWN or structural_break is True. |
| ImportError on `from regime_detection` | Package not installed in this Python env | Run `pip install -e /Users/user/regime-detection` |
| fathon install fails on macOS | No pre-built wheel for macOS/arm64 | Safe to ignore — pure numpy DFA fallback is used automatically. |
| HMM convergence warnings | Normal for small datasets | Suppressed internally. Does not affect output. |
