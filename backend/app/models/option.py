"""Option model for storing options contract data."""

from datetime import datetime, date
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, Date, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.db.database import Base


class Option(Base):
    """Option contract model with Greeks and pricing data."""
    
    __tablename__ = "options"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Contract identification
    symbol = Column(String, index=True, nullable=False)  # Option symbol
    underlying_symbol = Column(String, index=True, nullable=False)  # Stock symbol
    stock_id = Column(UUID(as_uuid=True), ForeignKey("stocks.id"))
    
    # Contract specifications
    strike = Column(Float, nullable=False)
    expiration = Column(Date, nullable=False)
    option_type = Column(String, nullable=False)  # call or put
    contract_size = Column(Integer, default=100)
    
    # Pricing data
    bid = Column(Float)
    ask = Column(Float)
    last_price = Column(Float)
    mark = Column(Float)  # Mid-price between bid and ask
    
    # Volume and open interest
    volume = Column(Integer)
    open_interest = Column(Integer)
    
    # Greeks
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    rho = Column(Float)
    
    # Implied volatility and pricing
    implied_volatility = Column(Float)
    theoretical_value = Column(Float)
    time_value = Column(Float)
    intrinsic_value = Column(Float)
    
    # Moneyness
    moneyness = Column(String)  # ITM, ATM, OTM
    
    # Change metrics
    change_amount = Column(Float)
    change_percent = Column(Float)
    
    # Additional metrics
    bid_size = Column(Integer)
    ask_size = Column(Integer)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    last_updated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("Stock", back_populates="options")
    
    # Indexes
    __table_args__ = (
        Index('idx_option_underlying_expiration', 'underlying_symbol', 'expiration'),
        Index('idx_option_strike_type', 'strike', 'option_type'),
        Index('idx_option_expiration', 'expiration'),
    )
    
    def __repr__(self):
        return f"<Option(symbol={self.symbol}, strike={self.strike}, type={self.option_type})>"
    
    @property
    def days_to_expiration(self):
        """Calculate days until expiration."""
        if self.expiration:
            return (self.expiration - date.today()).days
        return None
    
    @property
    def is_itm(self):
        """Check if option is in the money."""
        return self.moneyness == "ITM"
    
    @property
    def is_otm(self):
        """Check if option is out of the money."""
        return self.moneyness == "OTM"


class OptionChain(Base):
    """Aggregated option chain data for analysis."""
    
    __tablename__ = "option_chains"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    underlying_symbol = Column(String, index=True, nullable=False)
    
    # Chain metrics
    total_call_volume = Column(Integer)
    total_put_volume = Column(Integer)
    put_call_ratio = Column(Float)
    
    # Open interest
    total_call_oi = Column(Integer)
    total_put_oi = Column(Integer)
    
    # IV metrics
    iv_rank = Column(Float)  # 0-100 percentile
    iv_percentile = Column(Float)  # 0-100 percentile
    mean_iv = Column(Float)
    
    # Skew metrics
    put_call_skew = Column(Float)
    term_structure = Column(Float)
    
    # Timestamps
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<OptionChain(symbol={self.underlying_symbol}, pcr={self.put_call_ratio})>"


class OptionGreeksHistory(Base):
    """Historical Greeks data for options analysis."""
    
    __tablename__ = "option_greeks_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    option_id = Column(UUID(as_uuid=True), ForeignKey("option_contracts.id"))
    
    # Greeks snapshot
    timestamp = Column(DateTime, nullable=False)
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    rho = Column(Float)
    
    # Price data
    underlying_price = Column(Float)
    option_price = Column(Float)
    implied_volatility = Column(Float)
    
    # Indexes
    __table_args__ = (
        Index('idx_greeks_history_option_timestamp', 'option_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<OptionGreeksHistory(option_id={self.option_id}, timestamp={self.timestamp})>"
