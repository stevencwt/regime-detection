"""
regime_detection — Multi-Modal Market Regime Analysis Framework v3.1.

Public API:
    from regime_detection import RegimeManager

    manager = RegimeManager(config_path="my_config.yaml")
    manager.update(bar_dict, funding_rate=0.0003)
    regime = manager.get_current_regime()   # dict
    json_str = manager.get_json()           # JSON string
"""

from regime_detection.manager import RegimeManager
from regime_detection.schema import (
    ConsensusState,
    HMMLabel,
    LiquidityStatus,
    MarketType,
    RecommendedLogic,
    RegimeOutput,
    VolatilityRegime,
)

__version__ = "0.1.0"

__all__ = [
    "RegimeManager",
    "RegimeOutput",
    "ConsensusState",
    "MarketType",
    "VolatilityRegime",
    "HMMLabel",
    "LiquidityStatus",
    "RecommendedLogic",
]
