# Phase 5 — Integration Guide

## Overview

The regime-detection package is now functionally complete (185 tests passing).
This guide shows how to connect it to your existing bots with minimal code changes.

---

## Part 1: Connect to zpair (Hyperliquid Pairs Trading Bot)

### Step 1 — Install the package into zpair's environment

```bash
cd /Users/user/zpair
pip3 install -e /Users/user/regime-detection
```

Verify:
```bash
python3 -c "from regime_detection import RegimeManager; print('OK')"
```

### Step 2 — Drop the bridge file into zpair

Copy `regime_bridge.py` (provided in this delivery) to:

```
/Users/user/zpair/regime_bridge.py
```

This file is completely standalone — it does NOT modify any existing zpair code.

### Step 3 — Wire into vmain_v14s.py (3 small additions)

**Addition 1: Import (near the top, after other imports)**

Open `/Users/user/zpair/vmain_v14s.py` and add after the existing imports (around line 78):

```python
# ========================================================================
# REGIME DETECTION INTEGRATION (Zero Deletion Compliance: New code block)
# ========================================================================
try:
    from regime_bridge import RegimeBridge
    REGIME_BRIDGE_AVAILABLE = True
except ImportError:
    REGIME_BRIDGE_AVAILABLE = False
    print("[REGIME] regime_bridge not available — regime detection disabled")
# ========================================================================
```

**Addition 2: Initialize in TradingApplication.__init__() (after execution_engine init)**

Find the line `self._display_initialization_status()` (around line 785) and add BEFORE it:

```python
            # ========================================================================
            # REGIME DETECTION INITIALIZATION (Zero Deletion Compliance: New code block)
            # ========================================================================
            self.regime_bridge = None
            if REGIME_BRIDGE_AVAILABLE:
                try:
                    self.regime_bridge = RegimeBridge(
                        self.cfg,
                        self.telemetry_service,
                        strategy_type="pairs_trading",
                    )
                    logger.debug_print("Regime detection bridge initialized", "REGIME")
                except Exception as e:
                    logger.debug_print(f"Regime bridge init failed: {e}", "REGIME")
            # ========================================================================
```

**Addition 3: Update in main loop (after tele_wrapped is obtained)**

Find the section in the `run()` method where `tele_wrapped` is used (around line 2048):

```python
                        if barometer and tele_wrapped:
```

Add this block RIGHT AFTER that line (before the P&L update):

```python
                            # ========================================================
                            # REGIME DETECTION UPDATE (Zero Deletion: New code block)
                            # ========================================================
                            if self.regime_bridge:
                                try:
                                    regime = self.regime_bridge.update_from_telemetry(tele_wrapped)
                                    if regime.get("exit_mandate", False):
                                        logger.debug_print(
                                            f"REGIME EXIT MANDATE: {regime.get('consensus_state')}",
                                            "REGIME"
                                        )
                                except Exception as e:
                                    pass  # Non-critical — don't break trading loop
                            # ========================================================
```

That's it — 3 additions, zero modifications to existing code.

### Step 4 — Run zpair

```bash
cd /Users/user/zpair
python3 vmain_v14s.py --config config_SOL_MELANIA_5m1m.json
```

You'll see regime output in the console every ~60 seconds:

```
[REGIME] CHOP_NEUTRAL | PAIRS_MEAN_REVERSION | conf=0.72 vol=LOW_STABLE hmm=CHOP hurst=0.483 mandate=False
```

### Step 5 — Optional: Add regime to marketstate JSON

If you want regime data saved to your marketstate snapshots for the dashboard,
add this to the `create_base_marketstate()` method in vmain_v14s.py:

```python
            # Add regime detection data to marketstate
            if self.regime_bridge:
                marketstate["regime_detection"] = self.regime_bridge.get_regime()
```

---

## Part 2: Connect to moomoo bot (US Stocks — future)

The same pattern works for any bot. Here's how it would look for a moomoo/yfinance bot:

```python
from regime_detection import RegimeManager

# Initialize for US stock swing trading
regime_mgr = RegimeManager(
    market_type="US_STOCK",
    strategy_type="swing",
    market_class="us_stocks",
)

# In your main loop (wherever you get bar data from moomoo/yfinance):
def on_new_bar(bar_data, options_data=None):
    bar = {
        "timestamp": bar_data["timestamp"],
        "o": bar_data["open"],
        "h": bar_data["high"],
        "l": bar_data["low"],
        "c": bar_data["close"],
        "v": bar_data["volume"],
    }

    regime_mgr.update(
        bar=bar,
        options_snapshot=options_data,  # {"vanna": 0.3, "gamma": -0.5, "oi_skew": 1.2}
    )

    regime = regime_mgr.get_current_regime()
    print(f"Regime: {regime['recommended_logic']}")

    if regime["exit_mandate"]:
        close_all_positions()
```

For options income/selling strategies:

```python
regime_mgr = RegimeManager(
    market_type="US_STOCK",
    strategy_type="options_income",
    market_class="us_stocks",
)
```

---

## Part 3: Connect to future range/swing bots

```python
from regime_detection import RegimeManager

# Crypto scalping bot
regime_mgr = RegimeManager(
    market_type="CRYPTO_PERP",
    strategy_type="scalping",
    market_class="crypto",
)

# Or crypto range trading bot
regime_mgr = RegimeManager(
    market_type="CRYPTO_PERP",
    strategy_type="range_trading",
    market_class="crypto",
)

# The recommended_logic output tells you which playbook to activate:
#   SCALP_MEAN_REVERSION  → ultra-short, many trades, small targets
#   RANGE_TRADING         → buy low / sell high within channel
#   SWING_TREND_FOLLOW    → ride the trend
#   NO_TRADE              → stay flat
```

---

## Part 4: Custom config overrides

Create a YAML file to override any defaults without editing the package:

```yaml
# /Users/user/zpair/regime_config.yaml

# Tighter Hurst thresholds for your specific pair
hurst:
  range_min_hurst: 0.46
  range_max_hurst: 0.56
  trending_threshold: 0.62

# More sensitive CPD
cpd:
  penalty: 3.0
  recency_bars: 3

# Faster exit mandate
exit_mandate:
  grace_bars: 1
```

Then reference it in zpair's config.json:

```json
{
  "regime_detection": {
    "config_path": "regime_config.yaml"
  }
}
```

Or pass directly:

```python
regime_bridge = RegimeBridge(cfg, telemetry_service, strategy_type="pairs_trading")
# The bridge reads cfg["regime_detection"]["config_path"] automatically
```

---

## Part 5: Hot-reload (optional)

Edit your regime config YAML while the bot is running — changes take effect
without restart:

```python
# Add to your main loop (e.g., every 100 ticks):
if tick_count % 100 == 0 and self.regime_bridge:
    issues = self.regime_bridge.manager.reload_config()
    if issues:
        logger.debug_print(f"Regime config reload issues: {issues}", "REGIME")
```

Or use watchdog for automatic file-change detection:

```bash
pip3 install watchdog
```

```python
# watchdog_reload.py — drop into zpair/ (optional)
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigReloader(FileSystemEventHandler):
    def __init__(self, regime_bridge):
        self.bridge = regime_bridge

    def on_modified(self, event):
        if event.src_path.endswith(".yaml"):
            print(f"[HOT RELOAD] Config changed: {event.src_path}")
            self.bridge.manager.reload_config()

# Usage:
# observer = Observer()
# observer.schedule(ConfigReloader(self.regime_bridge), path=".", recursive=False)
# observer.start()
```

---

## Part 6: Reading regime output in your strategy

The regime output is available as a dict or JSON at any time:

```python
regime = self.regime_bridge.get_regime()

# Top-level fields
regime["consensus_state"]      # "CHOP_NEUTRAL", "BULL_PERSISTENT", etc.
regime["recommended_logic"]    # "PAIRS_MEAN_REVERSION", "SCALP_MEAN_REVERSION", etc.
regime["exit_mandate"]         # True → close all positions immediately
regime["confidence_score"]     # 0.0 – 1.0
regime["volatility_regime"]    # "LOW_STABLE", "EXPANDING", etc.

# Signal details
regime["signals"]["hmm_label"]          # "BULL", "BEAR", "CHOP"
regime["signals"]["hurst_dfa"]          # 0.0 – 1.0 (0.5 = random walk)
regime["signals"]["structural_break"]   # True/False
regime["signals"]["liquidity_status"]   # "CONSOLIDATION", "LIQUIDITY_TRAP"

# Crypto-specific
regime["signals"]["crypto_context"]["funding_bias"]   # "NEUTRAL", "EXTREME_POSITIVE"
regime["signals"]["crypto_context"]["funding_rate"]    # raw float

# Range hints (when in CHOP + scalp/range)
if "range_hints" in regime["signals"]:
    rh = regime["signals"]["range_hints"]
    rh["is_clean_range"]        # True = RANGE_TRADING conditions fully met
    rh["range_lower"]           # channel bottom
    rh["range_upper"]           # channel top
    rh["current_deviation_pct"] # how far from midpoint (%)
    rh["channel_type"]          # "Donchian_30"

# Quick convenience methods
self.regime_bridge.is_exit_mandated()       # bool
self.regime_bridge.get_recommended_logic()  # string
```

---

## Architecture Summary

```
zpair bot (vmain_v14s.py)
  │
  ├─ telemetry_service.get_market_data()  ← existing
  │       │
  │       ▼
  ├─ regime_bridge.update_from_telemetry()  ← NEW (3 lines added)
  │       │
  │       ├─ _extract_bar()         → OHLCV from tele_wrapped
  │       ├─ _get_funding_rate()    → from adapter.info.meta_and_asset_ctxs()
  │       ├─ _get_order_book_imbalance() → from streaming OrderbookStatsTracker
  │       ├─ _extract_spread_data() → spread series from buffer system
  │       │
  │       ▼
  │   RegimeManager.update()
  │       │
  │       ├─ HMM → BULL/BEAR/CHOP
  │       ├─ DFA Hurst → trending/mean-reverting
  │       ├─ CPD → structural break
  │       ├─ Volatility → LOW_STABLE/EXPANDING
  │       ├─ Consensus voting
  │       ├─ Recommended logic (scalp/range/swing/pairs)
  │       ├─ Range hints (Donchian boundaries)
  │       └─ Exit mandate
  │       │
  │       ▼
  │   regime dict (v3.1 JSON)
  │
  ├─ write_outputs() / strategy decisions  ← existing, can now use regime
  └─ execution_engine.process()            ← existing
```

---

## File Inventory (complete package)

```
/Users/user/regime-detection/          ← standalone package (own git repo)
├── pyproject.toml
├── README.md
├── config/
│   └── default_config.yaml            ← all v3.1 thresholds
├── src/regime_detection/
│   ├── __init__.py                    ← public API surface
│   ├── config.py                      ← YAML loading + validation
│   ├── schema.py                      ← enums + dataclasses + JSON
│   ├── signals.py                     ← HMM, Hurst, CPD, vol, liquidity
│   ├── processors.py                  ← crypto/options/pairs + consensus
│   └── recommendation.py             ← recommended logic + range hints + exit mandate
└── tests/
    ├── test_config.py                 ← 24 tests
    ├── test_schema.py                 ← 13 tests
    ├── test_manager.py                ← 25 tests
    ├── test_signals.py                ← 38 tests
    ├── test_processors.py             ← 42 tests
    └── test_recommendation.py         ← 43 tests
                                         ─────────
                                         185 total

/Users/user/zpair/                     ← your existing bot
├── regime_bridge.py                   ← NEW glue file (this delivery)
├── vmain_v14s.py                      ← 3 small additions (documented above)
└── (all existing files unchanged)
```
