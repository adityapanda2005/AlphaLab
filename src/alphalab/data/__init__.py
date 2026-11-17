"""Market data: the panel structure, the simulator and the loaders."""

from __future__ import annotations

from .loaders import (
    DEFAULT_UNIVERSE,
    DataSourceError,
    load_panel,
)
from .panel import CORE_FIELDS, MarketPanel, PanelError
from .synthetic import SECTORS, MarketConfig, generate_market, planted_effects

__all__ = [
    "CORE_FIELDS",
    "DEFAULT_UNIVERSE",
    "DataSourceError",
    "MarketConfig",
    "MarketPanel",
    "PanelError",
    "SECTORS",
    "generate_market",
    "load_panel",
    "planted_effects",
]
