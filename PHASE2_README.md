# Phase 2 Update Instructions

## Files in this delivery

### NEW files (add these):
- src/regime_detection/signals.py   — Core signal functions (HMM, Hurst, CPD, vol, liquidity, funding)
- tests/test_signals.py             — 38 tests with synthetic data

### UPDATED files (replace these):
- src/regime_detection/manager.py   — Stubs replaced with real signal calls

## How to apply

cd /Users/user/regime-detection

# Copy the new file
cp phase2_files/src/regime_detection/signals.py  src/regime_detection/signals.py

# Replace the updated file
cp phase2_files/src/regime_detection/manager.py  src/regime_detection/manager.py

# Copy the new test file
cp phase2_files/tests/test_signals.py  tests/test_signals.py

## Verify

python3 -m pytest tests/ -v
# Expect: 100 passed (62 Phase 1 + 38 Phase 2)

## Then commit

git add .
git commit -m "Phase 2: core signals — HMM, DFA Hurst, CPD, volatility, liquidity, funding — 100 tests"
git push
