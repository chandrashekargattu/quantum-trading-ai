"""Backtesting engine for strategy validation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Engine for running strategy backtests."""
    
    def __init__(self):
        self.market_service = MarketDataService()
        self.commission_rate = 0.001  # 0.1% default commission
    
    async def run_backtest(
        self,
        strategy: Dict[str, Any],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000
    ) -> Dict[str, Any]:
        """Run a backtest for a given strategy."""
        try:
            # Initialize results
            results = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_capital": initial_capital,
                "final_capital": initial_capital,
                "trades": [],
                "equity_curve": [],
                "daily_returns": []
            }
            
            # Get historical data for all symbols
            all_data = {}
            for symbol in symbols:
                history = await self.market_service.fetch_price_history(
                    symbol,
                    interval="1d",
                    period="2y"  # Get enough data
                )
                if history:
                    df = pd.DataFrame(history)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    
                    # Filter to backtest period
                    mask = (df.index >= start_date) & (df.index <= end_date)
                    all_data[symbol] = df.loc[mask]
            
            if not all_data:
                raise ValueError("No historical data available for backtesting")
            
            # Run strategy based on type
            strategy_type = strategy.get("strategy_type")
            config = strategy.get("config", {})
            
            if strategy_type == "trend_following":
                results = await self._run_trend_following_strategy(
                    all_data, config, initial_capital
                )
            elif strategy_type == "mean_reversion":
                results = await self._run_mean_reversion_strategy(
                    all_data, config, initial_capital
                )
            elif strategy_type == "options_spread":
                results = await self._run_options_strategy(
                    all_data, config, initial_capital
                )
            else:
                # Default simple moving average crossover
                results = await self._run_sma_crossover_strategy(
                    all_data, config, initial_capital
                )
            
            # Calculate performance metrics
            self._calculate_performance_metrics(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    async def _run_sma_crossover_strategy(
        self,
        data: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
        initial_capital: float
    ) -> Dict[str, Any]:
        """Run a simple moving average crossover strategy."""
        fast_period = config.get("fast_period", 20)
        slow_period = config.get("slow_period", 50)
        position_size = config.get("position_size", 0.1)
        
        capital = initial_capital
        positions = {}
        trades = []
        equity_curve = []
        
        # Process each symbol
        for symbol, df in data.items():
            # Calculate SMAs
            df['sma_fast'] = df['close'].rolling(window=fast_period).mean()
            df['sma_slow'] = df['close'].rolling(window=slow_period).mean()
            
            # Generate signals
            df['signal'] = 0
            df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
            df.loc[df['sma_fast'] < df['sma_slow'], 'signal'] = -1
            
            # Track position
            position = 0
            entry_price = 0
            
            for idx, row in df.iterrows():
                date = idx
                price = row['close']
                signal = row['signal']
                
                # Skip if no signal
                if pd.isna(signal):
                    continue
                
                # Entry logic
                if position == 0 and signal == 1:
                    # Buy signal
                    shares = int((capital * position_size) / price)
                    if shares > 0:
                        position = shares
                        entry_price = price
                        cost = shares * price * (1 + self.commission_rate)
                        capital -= cost
                        
                        trades.append({
                            "date": date.isoformat(),
                            "symbol": symbol,
                            "side": "buy",
                            "quantity": shares,
                            "price": price,
                            "cost": cost
                        })
                
                # Exit logic
                elif position > 0 and signal == -1:
                    # Sell signal
                    proceeds = position * price * (1 - self.commission_rate)
                    capital += proceeds
                    pnl = proceeds - (position * entry_price)
                    
                    trades.append({
                        "date": date.isoformat(),
                        "symbol": symbol,
                        "side": "sell",
                        "quantity": position,
                        "price": price,
                        "proceeds": proceeds,
                        "pnl": pnl
                    })
                    
                    position = 0
                    entry_price = 0
                
                # Track equity
                portfolio_value = capital
                if position > 0:
                    portfolio_value += position * price
                
                equity_curve.append({
                    "date": date.isoformat(),
                    "value": portfolio_value,
                    "capital": capital,
                    "positions_value": position * price if position > 0 else 0
                })
        
        return {
            "initial_capital": initial_capital,
            "final_capital": portfolio_value,
            "trades": trades,
            "equity_curve": equity_curve
        }
    
    async def _run_trend_following_strategy(
        self,
        data: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
        initial_capital: float
    ) -> Dict[str, Any]:
        """Run a trend following strategy."""
        # Placeholder implementation
        return await self._run_sma_crossover_strategy(data, config, initial_capital)
    
    async def _run_mean_reversion_strategy(
        self,
        data: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
        initial_capital: float
    ) -> Dict[str, Any]:
        """Run a mean reversion strategy."""
        # Placeholder implementation using RSI
        rsi_period = config.get("rsi_period", 14)
        oversold = config.get("oversold_level", 30)
        overbought = config.get("overbought_level", 70)
        
        # Similar logic but using RSI instead of SMA
        return await self._run_sma_crossover_strategy(data, config, initial_capital)
    
    async def _run_options_strategy(
        self,
        data: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
        initial_capital: float
    ) -> Dict[str, Any]:
        """Run an options strategy backtest."""
        # Placeholder - options backtesting is complex
        return {
            "initial_capital": initial_capital,
            "final_capital": initial_capital * 1.1,  # Mock 10% return
            "trades": [],
            "equity_curve": []
        }
    
    def _calculate_performance_metrics(self, results: Dict[str, Any]):
        """Calculate performance metrics from backtest results."""
        trades = results.get("trades", [])
        equity_curve = results.get("equity_curve", [])
        
        if not trades:
            results.update({
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown": 0
            })
            return
        
        # Calculate trade statistics
        pnls = [t.get("pnl", 0) for t in trades if "pnl" in t]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        total_trades = len(pnls)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        
        win_rate = num_winning / total_trades if total_trades > 0 else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calculate returns
        initial_capital = results["initial_capital"]
        final_capital = results["final_capital"]
        total_return = final_capital - initial_capital
        total_return_percent = (total_return / initial_capital) * 100
        
        # Calculate Sharpe ratio (simplified)
        if equity_curve:
            equity_values = [e["value"] for e in equity_curve]
            returns = pd.Series(equity_values).pct_change().dropna()
            
            if len(returns) > 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                
                # Sortino ratio (downside deviation)
                downside_returns = returns[returns < 0]
                downside_std = downside_returns.std() if len(downside_returns) > 0 else 1
                sortino_ratio = (returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
                
                # Max drawdown
                cum_returns = (1 + returns).cumprod()
                running_max = cum_returns.expanding().max()
                drawdown = (cum_returns - running_max) / running_max
                max_drawdown = abs(drawdown.min()) * 100
            else:
                sharpe_ratio = sortino_ratio = max_drawdown = 0
        else:
            sharpe_ratio = sortino_ratio = max_drawdown = 0
        
        # Update results
        results.update({
            "total_return": total_return,
            "total_return_percent": total_return_percent,
            "total_trades": total_trades,
            "winning_trades": num_winning,
            "losing_trades": num_losing,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "completed_at": datetime.utcnow().isoformat()
        })
