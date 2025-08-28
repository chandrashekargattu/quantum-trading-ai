"""Position and Transaction models for portfolio management."""

from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Boolean, Integer, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import enum

from app.db.database import Base


class TransactionType(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    DIVIDEND = "dividend"
    FEE = "fee"
    TRANSFER = "transfer"
    SPLIT = "split"
    INTEREST = "interest"


class AssetType(str, enum.Enum):
    STOCK = "stock"
    ETF = "etf"
    OPTION = "option"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"
    MUTUAL_FUND = "mutual_fund"


class Position(Base):
    __tablename__ = "portfolio_positions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String, nullable=False, index=True)
    
    # Position details
    quantity = Column(Float, nullable=False)
    average_cost = Column(Float, nullable=False)
    cost_basis = Column(Float, nullable=False)  # quantity * average_cost
    
    # Current market data
    current_price = Column(Float)
    market_value = Column(Float)
    
    # P&L
    unrealized_pnl = Column(Float)
    unrealized_pnl_percent = Column(Float)
    realized_pnl = Column(Float, default=0)
    
    # Asset information
    asset_type = Column(Enum(AssetType), default=AssetType.STOCK)
    asset_name = Column(String)
    sector = Column(String)
    industry = Column(String)
    
    # Risk management
    stop_loss = Column(Float)
    take_profit = Column(Float)
    
    # Position metadata
    is_long = Column(Boolean, default=True)
    is_margin = Column(Boolean, default=False)
    
    # Options specific (if applicable)
    option_type = Column(String)  # call/put
    strike_price = Column(Float)
    expiration_date = Column(DateTime)
    
    # Timestamps
    opened_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    closed_at = Column(DateTime)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")
    transactions = relationship("Transaction", back_populates="position")
    
    def __repr__(self):
        return f"<Position(symbol={self.symbol}, quantity={self.quantity}, value={self.market_value})>"


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    position_id = Column(UUID(as_uuid=True), ForeignKey("portfolio_positions.id"))
    
    # Transaction details
    transaction_type = Column(Enum(TransactionType), nullable=False)
    symbol = Column(String)  # Null for deposits/withdrawals
    quantity = Column(Float)
    price = Column(Float)
    amount = Column(Float, nullable=False)  # Total transaction amount
    
    # Fees and costs
    commission = Column(Float, default=0)
    fees = Column(Float, default=0)
    
    # Settlement
    trade_date = Column(DateTime, default=datetime.utcnow)
    settlement_date = Column(DateTime)
    
    # Additional details
    description = Column(String)
    reference_id = Column(String)  # External reference
    notes = Column(String)
    
    # Tax implications
    tax_lot_id = Column(String)
    wash_sale = Column(Boolean, default=False)
    
    # Status
    status = Column(String, default="completed")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="transactions")
    position = relationship("Position", back_populates="transactions")
    
    def __repr__(self):
        return f"<Transaction(type={self.transaction_type}, symbol={self.symbol}, amount={self.amount})>"


class PositionHistory(Base):
    __tablename__ = "position_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    position_id = Column(UUID(as_uuid=True), ForeignKey("portfolio_positions.id"), nullable=False)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    
    # Snapshot data
    date = Column(DateTime, nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    market_value = Column(Float, nullable=False)
    
    # P&L snapshot
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float)
    
    # Relationships
    position = relationship("Position")
    portfolio = relationship("Portfolio")
    
    def __repr__(self):
        return f"<PositionHistory(position_id={self.position_id}, date={self.date}, value={self.market_value})>"


class TaxLot(Base):
    __tablename__ = "tax_lots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    position_id = Column(UUID(as_uuid=True), ForeignKey("portfolio_positions.id"), nullable=False)
    
    # Lot details
    symbol = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    remaining_quantity = Column(Float, nullable=False)
    
    # Cost basis
    acquisition_date = Column(DateTime, nullable=False)
    acquisition_price = Column(Float, nullable=False)
    cost_basis = Column(Float, nullable=False)
    
    # Status
    is_closed = Column(Boolean, default=False)
    closed_date = Column(DateTime)
    
    # Tax information
    term = Column(String)  # short/long
    wash_sale_adjustment = Column(Float, default=0)
    
    # Relationships
    portfolio = relationship("Portfolio")
    position = relationship("Position")
    
    def __repr__(self):
        return f"<TaxLot(symbol={self.symbol}, quantity={self.quantity}, date={self.acquisition_date})>"
