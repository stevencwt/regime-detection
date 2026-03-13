"""
regime_detection.config — Configuration loading and validation.

Loads the default YAML config bundled with the package, then optionally
deep-merges a user-provided override file.  Supports hot-reload by
re-reading from disk on demand.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path to the default config shipped inside the package
# ---------------------------------------------------------------------------
_PACKAGE_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _PACKAGE_ROOT.parent.parent / "config" / "default_config.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base* (returns a new dict)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_default_config_path() -> Path:
    """Locate the bundled default_config.yaml.

    Tries two locations:
      1. ../../config/default_config.yaml relative to this file (editable install)
      2. A package-data fallback inside src/regime_detection/
    """
    if _DEFAULT_CONFIG_PATH.exists():
        return _DEFAULT_CONFIG_PATH

    # Fallback: same directory as this module (for wheel installs where
    # config/ is copied alongside the Python files)
    fallback = _PACKAGE_ROOT / "default_config.yaml"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Cannot find default_config.yaml. Searched:\n"
        f"  {_DEFAULT_CONFIG_PATH}\n"
        f"  {fallback}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_default_config() -> Dict[str, Any]:
    """Load and return the built-in default configuration."""
    path = _resolve_default_config_path()
    logger.debug("Loading default config from %s", path)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Default config is not a dict (got {type(cfg).__name__})")
    return cfg


def load_config(user_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load configuration with optional user override.

    Parameters
    ----------
    user_path : str | Path | None
        Path to a YAML file whose values override the defaults.
        If *None*, only the built-in defaults are returned.

    Returns
    -------
    dict
        Fully-merged configuration dictionary.
    """
    base = load_default_config()

    if user_path is None:
        logger.info("Using default config only (no user override)")
        return base

    user_path = Path(user_path)
    if not user_path.exists():
        raise FileNotFoundError(f"User config not found: {user_path}")

    logger.info("Loading user config override from %s", user_path)
    with open(user_path, "r", encoding="utf-8") as fh:
        overrides = yaml.safe_load(fh) or {}

    if not isinstance(overrides, dict):
        raise ValueError(f"User config is not a dict (got {type(overrides).__name__})")

    merged = _deep_merge(base, overrides)
    logger.debug("Config merged successfully (%d top-level keys)", len(merged))
    return merged


def validate_config(cfg: Dict[str, Any]) -> list[str]:
    """Run basic validation on a config dict.

    Returns a list of warning/error strings.  An empty list means the
    config passed all checks.
    """
    issues: list[str] = []

    # --- Required top-level sections ---
    required_sections = [
        "temporal_matrix", "hmm", "hurst", "cpd",
        "volatility", "liquidity", "funding",
        "range_detection", "exit_mandate",
    ]
    for section in required_sections:
        if section not in cfg:
            issues.append(f"MISSING required section: '{section}'")

    # --- HMM ---
    hmm = cfg.get("hmm", {})
    n_states = hmm.get("n_states", 0)
    if not isinstance(n_states, int) or n_states < 2:
        issues.append(f"hmm.n_states must be int >= 2 (got {n_states})")

    # --- Hurst ---
    hurst = cfg.get("hurst", {})
    rmin = hurst.get("range_min_hurst", 0)
    rmax = hurst.get("range_max_hurst", 1)
    trending = hurst.get("trending_threshold", 1)
    if not (0 < rmin < rmax < 1):
        issues.append(
            f"hurst range thresholds invalid: range_min={rmin}, range_max={rmax}"
        )
    if trending <= rmax:
        issues.append(
            f"hurst.trending_threshold ({trending}) should be > range_max_hurst ({rmax})"
        )

    # --- CPD ---
    cpd = cfg.get("cpd", {})
    penalty = cpd.get("penalty", 0)
    if not isinstance(penalty, (int, float)) or penalty <= 0:
        issues.append(f"cpd.penalty must be positive number (got {penalty})")

    # --- Volatility ---
    vol = cfg.get("volatility", {})
    band = vol.get("stable_band", [])
    if not (isinstance(band, list) and len(band) == 2 and band[0] < band[1]):
        issues.append(f"volatility.stable_band must be [low, high] (got {band})")

    # --- Exit mandate ---
    em = cfg.get("exit_mandate", {})
    grace = em.get("grace_bars", -1)
    if not isinstance(grace, int) or grace < 0:
        issues.append(f"exit_mandate.grace_bars must be non-negative int (got {grace})")

    if issues:
        for issue in issues:
            logger.warning("Config validation: %s", issue)
    else:
        logger.debug("Config validation passed — no issues found")

    return issues
