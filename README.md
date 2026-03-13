# regime-detection

**Multi-Modal Market Regime Analysis Framework v3.1**

A pure, reusable Python package for market regime detection. Classifies market states (trending, mean-reverting, choppy) and recommends execution logic (scalping, range trading, swing, pairs, options) via a hybrid consensus engine.

## Quick Start

```python
from regime_detection import RegimeManager

manager = RegimeManager(
    config_path="my_config.yaml",   # optional override
    market_type="CRYPTO_PERP",
    strategy_type="scalping",
    market_class="crypto",
)

# Feed bars from your existing bot
manager.update(
    bar={"timestamp": 1710000000, "o": 150.1, "h": 151.2, "l": 149.8, "c": 150.5, "v": 42000},
    funding_rate=0.0003,
    order_book_imbalance=-0.12,
)

# Read regime
regime = manager.get_current_regime()  # dict
print(regime["recommended_logic"])     # e.g. "SCALP_MEAN_REVERSION"
print(regime["exit_mandate"])          # True → close all positions

# Or get JSON
print(manager.get_json())
```

## Installation (editable)

```bash
cd regime-detection
pip install -e ".[dev]"
```

Then add to any bot:
```bash
# from your bot repo
pip install -e /path/to/regime-detection
```

## Architecture

This package is **pure computation** — no exchange connectors, no API clients. Data flows in via `.update()`, regime flows out via `.get_json()`.

```
Your Bot (zpair / moomoo / future)
  │
  │  bar, funding, imbalance, options, spread
  ▼
RegimeManager.update()
  │
  ├─ HMM (hmmlearn)        → BULL / BEAR / CHOP
  ├─ DFA Hurst (fathon)    → trending vs mean-reverting
  ├─ CPD (ruptures)        → structural break detection
  ├─ Volatility classifier → LOW_STABLE / EXPANDING / etc.
  ├─ Liquidity heuristic   → CONSOLIDATION / TRAP
  ├─ Market processors     → crypto context, options context, pairs context
  │
  └─ Consensus Engine
       │
       ▼
  RegimeOutput (v3.1 JSON)
    ├─ consensus_state
    ├─ recommended_logic  (SCALP / RANGE / SWING / PAIRS / OPTIONS / NO_TRADE)
    ├─ exit_mandate       (regime shift → force close)
    └─ signals + range_hints
```

## Development

```bash
pip install -e ".[dev]"
pytest -v
```

## Phases

- **Phase 1** ✅ Package skeleton, config, schema, manager API
- **Phase 2** Core signals (HMM, Hurst, CPD, volatility)
- **Phase 3** Market processors (crypto, options, pairs)
- **Phase 4** Consensus engine, range vs scalp, exit mandate
- **Phase 5** Integration with zpair + moomoo bots
