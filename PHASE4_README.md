# Phase 4 Update Instructions

## Files in this delivery

### NEW files (add these):
- `src/regime_detection/recommendation.py` — Recommended logic engine, range hints, range persistence, exit mandate
- `tests/test_recommendation.py` — 43 tests for all Phase 4 logic

### UPDATED files (replace these):
- `src/regime_detection/manager.py` — Phase 4 stubs replaced with real calls, added state tracking
- `tests/test_manager.py` — Updated OutputAfterWarmup to test real Phase 4 values
- `tests/test_processors.py` — Updated 2 stub assertions to validate real Phase 4 output

## How to apply

```bash
cd /Users/user/regime-detection

# NEW files
cp <download_path>/recommendation.py  src/regime_detection/recommendation.py
cp <download_path>/test_recommendation.py  tests/test_recommendation.py

# UPDATED files (replace)
cp <download_path>/manager.py  src/regime_detection/manager.py
cp <download_path>/test_manager.py  tests/test_manager.py
cp <download_path>/test_processors.py  tests/test_processors.py
```

## Verify

```bash
python3 -m pytest tests/ -v
```

Expected: **185 passed** (37 config+schema + 25 manager + 38 signals + 42 processors + 43 recommendation)

## Then commit

```bash
git add .
git commit -m "Phase 4: recommended logic, range vs scalp, range hints, exit mandate — 185 tests"
git push
```

## What Phase 4 added

### recommendation.py (416 lines) — 4 functions

| Function | What it does |
|---|---|
| `determine_recommended_logic()` | Maps consensus + signals → v3.1 recommended_logic. Handles safety gates (UNKNOWN/TRANSITION/break→NO_TRADE), strategy overrides (pairs→PAIRS_MEAN_REVERSION, options_income→OPTIONS_INCOME), trending→SWING_TREND_FOLLOW, and CHOP→scalp vs range |
| `_classify_chop_sub_regime()` | The key v3.1 distinction within CHOP_NEUTRAL: RANGE_TRADING (Hurst 0.48–0.58 + LOW_STABLE vol + CONSOLIDATION + persistence ≥10 bars) vs SCALP_MEAN_REVERSION (Hurst <0.48 + CONSOLIDATION) vs NO_TRADE |
| `compute_range_hints()` | Donchian/Keltner channel boundaries, current deviation %, is_clean_range flag. Only computed when CHOP + active mean-reversion logic |
| `compute_range_persistence()` | Counts consecutive recent bars inside the channel (scans backward from latest) |
| `evaluate_exit_mandate()` | Grace-period regime shift detection + 3 immediate triggers: CPD break, Hurst above trending (from CHOP), EXPANDING volatility (from CHOP). Returns (mandate_bool, updated_counter) |

### v3.1 Activation Rules Implemented

| Condition | Result |
|---|---|
| UNKNOWN / TRANSITION / structural break / None Hurst | NO_TRADE |
| pairs_trading + CHOP | PAIRS_MEAN_REVERSION |
| pairs_trading + trending | NO_TRADE |
| options_income + stable vol | OPTIONS_INCOME |
| options_speculative + trending or expanding vol | OPTIONS_SPECULATIVE |
| BULL/BEAR_PERSISTENT | SWING_TREND_FOLLOW |
| CHOP + Hurst <0.48 + CONSOLIDATION | SCALP_MEAN_REVERSION |
| CHOP + Hurst 0.48–0.58 + LOW_STABLE + persistence ≥10 | RANGE_TRADING |
| CHOP + Hurst 0.48–0.58 but persistence not met | SCALP_MEAN_REVERSION (interim) |
| CHOP + LIQUIDITY_TRAP | NO_TRADE |

### Exit Mandate Logic

| Trigger | Grace Period? |
|---|---|
| CPD structural break | Immediate |
| Hurst ≥ trending_threshold (was CHOP) | Immediate |
| Volatility → EXPANDING (was CHOP) | Immediate |
| Consensus state change (e.g. CHOP→BULL) | grace_bars confirmation required |
| exit_mandate.enabled = false | Never triggers |

### What's complete after Phase 4

The regime detection module is now **functionally complete**. The entire v3.1 spec pipeline works end-to-end:

```
bar data → HMM + Hurst + CPD + Volatility + Liquidity
         → Market processors (crypto/options/pairs)
         → Consensus voting
         → Recommended logic (scalp vs range vs swing vs pairs vs options)
         → Range hints (Donchian/Keltner boundaries)
         → Exit mandate (grace period + immediate triggers)
         → v3.1 JSON output
```

Phase 5 (next) will provide integration instructions for connecting to zpair and moomoo bots.
