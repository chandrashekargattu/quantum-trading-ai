"""Options pricing service using Black-Scholes model and other pricing methods."""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)


class OptionsPricingService:
    """Service for pricing options using various models."""
    
    def __init__(self):
        """Initialize the options pricing service."""
        self.risk_free_rate = 0.05  # Default 5% risk-free rate
        
    def calculate_greeks(
        self,
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: Optional[float] = None,
        option_type: str = "call",
        dividend_yield: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate option Greeks using Black-Scholes model.
        
        Args:
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility (annualized)
            risk_free_rate: Risk-free interest rate
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield of the underlying
            
        Returns:
            Dictionary containing delta, gamma, theta, vega, and rho
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        # Prevent division by zero
        if time_to_expiry <= 0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "rho": 0.0
            }
            
        # Calculate d1 and d2
        d1 = (np.log(underlying_price / strike_price) + 
              (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        # Calculate Greeks
        if option_type.lower() == "call":
            delta = np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
            rho = strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100
        else:  # put
            delta = -np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
            rho = -strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100
            
        # Greeks common to both calls and puts
        gamma = np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) / \
                (underlying_price * volatility * np.sqrt(time_to_expiry))
        
        theta_common = -(underlying_price * volatility * np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1)) / \
                       (2 * np.sqrt(time_to_expiry))
        
        if option_type.lower() == "call":
            theta = (theta_common - 
                    risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) +
                    dividend_yield * underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)) / 365
        else:  # put
            theta = (theta_common + 
                    risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) -
                    dividend_yield * underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)) / 365
            
        vega = underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100
        
        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 4),
            "theta": round(theta, 4),
            "vega": round(vega, 4),
            "rho": round(rho, 4)
        }
    
    def calculate_black_scholes_price(
        self,
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: Optional[float] = None,
        option_type: str = "call",
        dividend_yield: float = 0.0
    ) -> float:
        """
        Calculate option price using Black-Scholes model.
        
        Args:
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility (annualized)
            risk_free_rate: Risk-free interest rate
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield of the underlying
            
        Returns:
            Theoretical option price
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        if time_to_expiry <= 0:
            # Option has expired
            if option_type.lower() == "call":
                return max(0, underlying_price - strike_price)
            else:
                return max(0, strike_price - underlying_price)
                
        # Calculate d1 and d2
        d1 = (np.log(underlying_price / strike_price) + 
              (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        if option_type.lower() == "call":
            price = (underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1) - 
                    strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
        else:  # put
            price = (strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - 
                    underlying_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1))
                    
        return round(price, 2)
    
    def calculate_implied_volatility(
        self,
        option_price: float,
        underlying_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: Optional[float] = None,
        option_type: str = "call",
        dividend_yield: float = 0.0
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            option_price: Current market price of the option
            underlying_price: Current price of the underlying asset
            strike_price: Strike price of the option
            time_to_expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield of the underlying
            
        Returns:
            Implied volatility or None if calculation fails
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        # Initial guess
        volatility = 0.3
        
        # Newton-Raphson iteration
        max_iterations = 100
        tolerance = 1e-5
        
        for _ in range(max_iterations):
            price = self.calculate_black_scholes_price(
                underlying_price, strike_price, time_to_expiry,
                volatility, risk_free_rate, option_type, dividend_yield
            )
            
            # Calculate vega
            greeks = self.calculate_greeks(
                underlying_price, strike_price, time_to_expiry,
                volatility, risk_free_rate, option_type, dividend_yield
            )
            vega = greeks["vega"] * 100  # Convert back from percentage
            
            price_diff = option_price - price
            
            if abs(price_diff) < tolerance:
                return round(volatility, 4)
                
            if vega == 0:
                return None
                
            volatility = volatility + price_diff / vega
            
            # Ensure volatility stays positive
            if volatility <= 0:
                volatility = 0.01
                
        return None
    
    def calculate_payoff(
        self,
        option_type: str,
        strike_price: float,
        underlying_price: float,
        premium: float,
        position: str = "long"
    ) -> float:
        """
        Calculate option payoff at expiration.
        
        Args:
            option_type: 'call' or 'put'
            strike_price: Strike price of the option
            underlying_price: Price of underlying at expiration
            premium: Premium paid/received for the option
            position: 'long' or 'short'
            
        Returns:
            Net payoff including premium
        """
        if option_type.lower() == "call":
            intrinsic_value = max(0, underlying_price - strike_price)
        else:  # put
            intrinsic_value = max(0, strike_price - underlying_price)
            
        if position.lower() == "long":
            return intrinsic_value - premium
        else:  # short
            return premium - intrinsic_value
    
    def calculate_breakeven(
        self,
        option_type: str,
        strike_price: float,
        premium: float,
        position: str = "long"
    ) -> float:
        """
        Calculate breakeven price for an option position.
        
        Args:
            option_type: 'call' or 'put'
            strike_price: Strike price of the option
            premium: Premium paid/received for the option
            position: 'long' or 'short'
            
        Returns:
            Breakeven price
        """
        if option_type.lower() == "call":
            if position.lower() == "long":
                return strike_price + premium
            else:  # short call
                return strike_price + premium
        else:  # put
            if position.lower() == "long":
                return strike_price - premium
            else:  # short put
                return strike_price - premium
    
    def calculate_time_to_expiry_years(self, expiration_date: date) -> float:
        """
        Calculate time to expiry in years from today.
        
        Args:
            expiration_date: Option expiration date
            
        Returns:
            Time to expiry in years
        """
        today = date.today()
        days_to_expiry = (expiration_date - today).days
        return max(0, days_to_expiry / 365.25)
    
    def analyze_option_strategy(
        self,
        strategy_type: str,
        legs: list,
        underlying_price: float
    ) -> Dict[str, any]:
        """
        Analyze multi-leg option strategies.
        
        Args:
            strategy_type: Type of strategy (e.g., 'iron_condor', 'butterfly')
            legs: List of option legs with details
            underlying_price: Current underlying price
            
        Returns:
            Strategy analysis including max profit, max loss, breakeven points
        """
        total_cost = 0
        payoffs = []
        
        # Calculate total cost and individual payoffs
        for leg in legs:
            if leg.get("action") == "buy":
                total_cost += leg.get("premium", 0) * leg.get("quantity", 1)
            else:  # sell
                total_cost -= leg.get("premium", 0) * leg.get("quantity", 1)
        
        # Calculate payoff at various underlying prices
        price_range = np.linspace(
            underlying_price * 0.8,
            underlying_price * 1.2,
            100
        )
        
        strategy_payoffs = []
        for price in price_range:
            total_payoff = -total_cost
            for leg in legs:
                payoff = self.calculate_payoff(
                    leg.get("option_type"),
                    leg.get("strike_price"),
                    price,
                    leg.get("premium"),
                    "long" if leg.get("action") == "buy" else "short"
                )
                total_payoff += payoff * leg.get("quantity", 1)
            strategy_payoffs.append(total_payoff)
        
        # Find max profit, max loss, and breakeven points
        max_profit = max(strategy_payoffs)
        max_loss = min(strategy_payoffs)
        
        # Find breakeven points
        breakeven_points = []
        for i in range(1, len(strategy_payoffs)):
            if (strategy_payoffs[i-1] < 0 and strategy_payoffs[i] >= 0) or \
               (strategy_payoffs[i-1] >= 0 and strategy_payoffs[i] < 0):
                # Linear interpolation for more accurate breakeven
                breakeven = price_range[i-1] + (price_range[i] - price_range[i-1]) * \
                           (-strategy_payoffs[i-1] / (strategy_payoffs[i] - strategy_payoffs[i-1]))
                breakeven_points.append(round(breakeven, 2))
        
        return {
            "strategy_type": strategy_type,
            "total_cost": round(total_cost, 2),
            "max_profit": round(max_profit, 2),
            "max_loss": round(max_loss, 2),
            "breakeven_points": breakeven_points,
            "current_underlying": underlying_price
        }
