# Phase 3 Update Instructions

## Files in this delivery

### NEW files (add these):
- src/regime_detection/processors.py   — Market processors (crypto, options, pairs) + consensus voting engine
- tests/test_processors.py             — 42 tests for processors + consensus + integration

### UPDATED files (replace these):
- src/regime_detection/manager.py      — Phase 3 stubs replaced with real processor + consensus calls
- tests/test_manager.py                — Updated to test real consensus output (was testing stubs before)

## How to apply

```bash
cd /Users/user/regime-detection

# Copy new files
cp <download_path>/processors.py  src/regime_detection/processors.py
cp <download_path>/test_processors.py  tests/test_processors.py

# Replace updated files
cp <download_path>/manager.py  src/regime_detection/manager.py
cp <download_path>/test_manager.py  tests/test_manager.py
```

## Verify

```bash
python3 -m pytest tests/ -v
```

Expected: **142 passed** (37 config+schema + 25 manager + 38 signals + 42 processors)

Note: HMM fitting tests take 2-4 minutes total. The fast tests (config, schema, processors unit) finish in seconds.

## Then commit

```bash
git add .
git commit -m "Phase 3: market processors (crypto, options, pairs) + consensus voting — 142 tests"
git push
```

## What Phase 3 added

### processors.py (372 lines)
| Function | What it does |
|---|---|
| `process_crypto()` | Builds CryptoContext from funding rate + order-book imbalance |
| `process_options()` | Builds OptionsContext from vanna, gamma flip, OI skew snapshot |
| `process_pairs()` | Builds PairsContext — computes spread Hurst via DFA, reads half-life + cointegration p-value |
| `vote_consensus()` | Consensus voting engine — combines HMM label, Hurst, CPD, vol regime into ConsensusState + confidence |

### Consensus voting rules (v3.1 spec Section 4)
- BULL_PERSISTENT: HMM=BULL + Hurst >= 0.60 + no break + vol not EXPANDING
- BEAR_PERSISTENT: HMM=BEAR + Hurst >= 0.60 + no break + vol not EXPANDING
- CHOP_NEUTRAL: Hurst < 0.60 (regardless of HMM label)
- TRANSITION: structural break OR (EXPANDING vol + trending Hurst) OR signal conflict
- UNKNOWN: insufficient data

### Missing data handling
- Any None input → that field shows UNKNOWN, other fields still computed
- Partial options/pairs snapshots gracefully skip missing keys
- Invalid types (strings where floats expected) caught and logged, no crash

### What's still stubbed (Phase 4)
- `recommended_logic` → always NO_TRADE
- `range_hints` → always None
- `exit_mandate` → always False
