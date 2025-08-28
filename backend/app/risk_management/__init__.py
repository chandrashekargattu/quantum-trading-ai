"""Advanced risk management with tail risk modeling and dynamic hedging."""

from .advanced_risk_manager import (
    AdvancedRiskManager,
    RiskMetrics,
    TailRiskEvent,
    ExtremeValueAnalyzer,
    CopulaRiskModeler,
    RegimeSwitchingRiskModel,
    DynamicHedgingEngine,
    StressTestingEngine
)

__all__ = [
    "AdvancedRiskManager",
    "RiskMetrics",
    "TailRiskEvent",
    "ExtremeValueAnalyzer",
    "CopulaRiskModeler",
    "RegimeSwitchingRiskModel",
    "DynamicHedgingEngine",
    "StressTestingEngine"
]
