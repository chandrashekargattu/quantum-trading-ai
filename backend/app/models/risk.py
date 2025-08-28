"""Risk management models."""

from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Enum, Boolean, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import enum

from app.db.database import Base


class RiskLimitType(str, enum.Enum):
    VAR = "var"
    LEVERAGE = "leverage"
    CONCENTRATION = "concentration"
    SECTOR = "sector"
    LOSS_DAILY = "loss_daily"
    LOSS_MONTHLY = "loss_monthly"
    POSITION_SIZE = "position_size"
    DRAWDOWN = "drawdown"


class AlertSeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RiskMetrics(Base):
    __tablename__ = "risk_metrics_detailed"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    
    # VaR metrics
    var_95_1d = Column(Float)  # 95% 1-day VaR
    var_99_1d = Column(Float)  # 99% 1-day VaR
    var_95_10d = Column(Float)  # 95% 10-day VaR
    cvar_95 = Column(Float)  # Conditional VaR (Expected Shortfall)
    
    # Portfolio risk metrics
    portfolio_beta = Column(Float)
    portfolio_volatility = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    
    # Concentration metrics
    herfindahl_index = Column(Float)
    largest_position_weight = Column(Float)
    top_5_concentration = Column(Float)
    
    # Leverage metrics
    gross_leverage = Column(Float)
    net_leverage = Column(Float)
    
    # Liquidity metrics
    liquidity_score = Column(Float)
    days_to_liquidate = Column(Float)
    
    # Drawdown metrics
    current_drawdown = Column(Float)
    max_drawdown = Column(Float)
    drawdown_duration = Column(Integer)  # Days
    
    # Greeks (for options portfolios)
    portfolio_delta = Column(Float)
    portfolio_gamma = Column(Float)
    portfolio_vega = Column(Float)
    portfolio_theta = Column(Float)
    
    # Timestamp
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio")


class RiskLimits(Base):
    __tablename__ = "risk_limits"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    limit_type = Column(Enum(RiskLimitType), nullable=False)
    
    # Limit values
    limit_value = Column(Float, nullable=False)
    warning_threshold = Column(Float)  # Percentage of limit (e.g., 0.8 = 80%)
    
    # Current status
    current_value = Column(Float)
    is_breached = Column(Boolean, default=False)
    breach_count = Column(Integer, default=0)
    
    # Metadata
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_breach_at = Column(DateTime)
    
    # Relationships
    portfolio = relationship("Portfolio")
    alerts = relationship("RiskAlert", back_populates="risk_limit")


class RiskAlert(Base):
    __tablename__ = "risk_alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    risk_limit_id = Column(UUID(as_uuid=True), ForeignKey("risk_limits.id"))
    
    # Alert details
    alert_type = Column(String, nullable=False)
    severity = Column(Enum(AlertSeverity), nullable=False)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    
    # Alert values
    current_value = Column(Float)
    threshold_value = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    acknowledged_at = Column(DateTime)
    
    # Resolution
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio")
    risk_limit = relationship("RiskLimits", back_populates="alerts")
    acknowledged_user = relationship("User")


class StressTestScenario(Base):
    __tablename__ = "stress_test_scenarios"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(Text)
    scenario_type = Column(String, nullable=False)  # historical, hypothetical, custom
    
    # Scenario parameters (stored as JSON)
    market_shocks = Column(JSON)  # {asset: shock_percentage}
    correlation_adjustments = Column(JSON)
    volatility_multipliers = Column(JSON)
    
    # Reference data
    historical_date = Column(DateTime)  # For historical scenarios
    
    # Status
    is_active = Column(Boolean, default=True)
    is_regulatory = Column(Boolean, default=False)  # Required by regulators
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StressTestResult(Base):
    __tablename__ = "stress_test_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    scenario_id = Column(UUID(as_uuid=True), ForeignKey("stress_test_scenarios.id"), nullable=False)
    
    # Results
    portfolio_value_before = Column(Float, nullable=False)
    portfolio_value_after = Column(Float, nullable=False)
    absolute_loss = Column(Float)
    percentage_loss = Column(Float)
    
    # Position-level impacts (stored as JSON)
    position_impacts = Column(JSON)
    
    # Risk metrics under stress
    stressed_var = Column(Float)
    stressed_volatility = Column(Float)
    
    # Timestamp
    tested_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio")
    scenario = relationship("StressTestScenario")


class RiskReport(Base):
    __tablename__ = "risk_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    report_type = Column(String, nullable=False)  # daily, weekly, monthly, ad-hoc
    
    # Report content
    executive_summary = Column(Text)
    risk_metrics = Column(JSON)
    limit_utilization = Column(JSON)
    stress_test_summary = Column(JSON)
    recommendations = Column(JSON)
    
    # File storage
    file_path = Column(String)
    file_format = Column(String)  # pdf, excel, json
    
    # Period covered
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Status
    is_final = Column(Boolean, default=False)
    approved_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    approved_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio")
    approver = relationship("User")


class RiskModel(Base):
    __tablename__ = "risk_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # var, factor, copula, etc.
    version = Column(String, nullable=False)
    
    # Model parameters
    parameters = Column(JSON)
    
    # Validation metrics
    backtest_violations = Column(Integer)
    backtest_periods = Column(Integer)
    kupiec_test_statistic = Column(Float)
    kupiec_test_pvalue = Column(Float)
    christoffersen_test_statistic = Column(Float)
    christoffersen_test_pvalue = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_validated = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    validated_at = Column(DateTime)
    
    # Relationships
    validation_results = relationship("ModelValidationResult", back_populates="risk_model")


class ModelValidationResult(Base):
    __tablename__ = "model_validation_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    risk_model_id = Column(UUID(as_uuid=True), ForeignKey("risk_models.id"), nullable=False)
    
    # Validation details
    validation_type = Column(String, nullable=False)  # backtest, stress_test, benchmark
    test_period_start = Column(DateTime, nullable=False)
    test_period_end = Column(DateTime, nullable=False)
    
    # Results
    test_statistic = Column(Float)
    p_value = Column(Float)
    passed = Column(Boolean)
    
    # Detailed results (stored as JSON)
    detailed_results = Column(JSON)
    
    # Timestamp
    validated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    risk_model = relationship("RiskModel", back_populates="validation_results")
