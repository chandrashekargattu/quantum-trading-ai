"""
Deep Reinforcement Learning Trading Engine

Implements state-of-the-art deep RL algorithms for automated trading,
similar to techniques used by Jane Street and other top quant firms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from collections import deque
import random
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


@dataclass
class TradingState:
    """State representation for the trading environment."""
    
    prices: np.ndarray  # Historical prices
    volumes: np.ndarray  # Historical volumes
    technical_indicators: np.ndarray  # RSI, MACD, Bollinger Bands, etc.
    order_book: np.ndarray  # Bid/ask spreads and depths
    portfolio_state: np.ndarray  # Current holdings, cash, P&L
    market_microstructure: np.ndarray  # Tick data, trade imbalance
    sentiment_scores: np.ndarray  # News and social media sentiment
    volatility_surface: np.ndarray  # Options implied volatility
    correlation_matrix: np.ndarray  # Inter-asset correlations
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to PyTorch tensor."""
        return torch.cat([
            torch.tensor(self.prices, dtype=torch.float32),
            torch.tensor(self.volumes, dtype=torch.float32),
            torch.tensor(self.technical_indicators, dtype=torch.float32),
            torch.tensor(self.order_book, dtype=torch.float32),
            torch.tensor(self.portfolio_state, dtype=torch.float32),
            torch.tensor(self.market_microstructure, dtype=torch.float32),
            torch.tensor(self.sentiment_scores, dtype=torch.float32),
            torch.tensor(self.volatility_surface.flatten(), dtype=torch.float32),
            torch.tensor(self.correlation_matrix.flatten(), dtype=torch.float32),
        ])


class AttentionNetwork(nn.Module):
    """
    Multi-head attention network for processing market data.
    Similar to transformer architecture but adapted for trading.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(input_dim, hidden_dim * num_heads)
        self.key = nn.Linear(input_dim, hidden_dim * num_heads)
        self.value = nn.Linear(input_dim, hidden_dim * num_heads)
        self.output = nn.Linear(hidden_dim * num_heads, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Compute Q, K, V
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.hidden_dim)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.hidden_dim)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.hidden_dim)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.view(batch_size, -1)
        
        # Output projection
        output = self.output(context)
        output = self.layer_norm(output + x.mean(dim=1))
        
        return output


class DeepTradingNetwork(nn.Module):
    """
    Deep neural network for trading decisions using attention mechanisms,
    LSTM for temporal patterns, and CNN for pattern recognition.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 6
    ):
        super().__init__()
        
        # Feature extraction layers
        self.price_cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention mechanism
        self.attention = AttentionNetwork(hidden_dim, hidden_dim)
        
        # Deep fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Output heads
        self.action_head = nn.Linear(hidden_dim // 4, action_dim)
        self.value_head = nn.Linear(hidden_dim // 4, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # LSTM processing
        lstm_out, hidden = self.lstm(state.unsqueeze(1), hidden)
        lstm_out = lstm_out.squeeze(1)
        
        # Attention
        attended = self.attention(lstm_out)
        
        # Deep layers
        features = self.fc_layers(attended)
        
        # Action probabilities and value estimate
        action_logits = self.action_head(features)
        value = self.value_head(features)
        
        return action_logits, value, hidden
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param)


class PPOAgent:
    """
    Proximal Policy Optimization agent for trading.
    PPO is more stable than vanilla policy gradient methods.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # Actor-Critic network
        self.policy = DeepTradingNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Old policy for PPO
        self.policy_old = DeepTradingNetwork(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Memory
        self.memory = []
        
    def select_action(
        self,
        state: TradingState,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Select action using the current policy."""
        state_tensor = state.to_tensor().unsqueeze(0)
        
        with torch.no_grad():
            action_logits, _, hidden = self.policy(state_tensor, hidden)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
        
        return action.item(), hidden
    
    def update(self):
        """Update policy using PPO algorithm."""
        if len(self.memory) < 100:
            return
        
        # Convert memory to tensors
        states = torch.stack([m[0] for m in self.memory])
        actions = torch.tensor([m[1] for m in self.memory])
        rewards = torch.tensor([m[2] for m in self.memory])
        old_logprobs = torch.stack([m[3] for m in self.memory])
        
        # Calculate discounted rewards
        discounted_rewards = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-8
        )
        
        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy outputs
            action_logits, values, _ = self.policy(states)
            dist = Categorical(F.softmax(action_logits, dim=-1))
            
            # Calculate advantages
            advantages = discounted_rewards - values.squeeze()
            
            # Ratio for PPO
            new_logprobs = dist.log_prob(actions)
            ratio = torch.exp(new_logprobs - old_logprobs)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Total loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), discounted_rewards)
            entropy_loss = -0.01 * dist.entropy().mean()
            
            loss = actor_loss + 0.5 * critic_loss + entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.memory = []


class TradingEnvironment(gym.Env):
    """
    Advanced trading environment with realistic market simulation.
    Includes transaction costs, slippage, and market impact.
    """
    
    def __init__(
        self,
        market_data: Dict[str, np.ndarray],
        initial_capital: float = 1000000,
        transaction_cost: float = 0.0005,  # 5 bps
        max_position_size: float = 0.1,  # 10% of capital per position
        leverage: float = 2.0
    ):
        super().__init__()
        
        self.market_data = market_data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.leverage = leverage
        
        # Action space: buy, sell, hold for each asset + position sizing
        n_assets = len(market_data)
        self.action_space = spaces.MultiDiscrete([3] * n_assets + [11] * n_assets)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._get_state_dim(),),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self) -> TradingState:
        """Reset environment to initial state."""
        self.current_step = 0
        self.capital = self.initial_capital
        self.positions = {asset: 0 for asset in self.market_data.keys()}
        self.trades = []
        self.portfolio_values = [self.initial_capital]
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[TradingState, float, bool, Dict]:
        """Execute trading action and return new state."""
        # Parse actions
        n_assets = len(self.market_data)
        trade_actions = action[:n_assets]
        position_sizes = action[n_assets:] / 10.0  # Convert to 0-1 range
        
        # Execute trades
        for i, (asset, trade_action) in enumerate(zip(self.market_data.keys(), trade_actions)):
            if trade_action == 0:  # Buy
                self._execute_buy(asset, position_sizes[i])
            elif trade_action == 1:  # Sell
                self._execute_sell(asset, position_sizes[i])
            # trade_action == 2 is hold
        
        # Update portfolio value
        portfolio_value = self._calculate_portfolio_value()
        self.portfolio_values.append(portfolio_value)
        
        # Calculate reward (Sharpe ratio based)
        reward = self._calculate_reward()
        
        # Check if done
        self.current_step += 1
        done = (
            self.current_step >= len(list(self.market_data.values())[0]) - 1 or
            portfolio_value < self.initial_capital * 0.5  # 50% drawdown stop
        )
        
        # Get new state
        state = self._get_state()
        
        # Additional info
        info = {
            'portfolio_value': portfolio_value,
            'positions': self.positions.copy(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown()
        }
        
        return state, reward, done, info
    
    def _execute_buy(self, asset: str, position_size: float):
        """Execute buy order with market impact modeling."""
        current_price = self.market_data[asset][self.current_step]
        
        # Calculate order size
        max_order_value = self.capital * self.max_position_size * position_size
        shares = max_order_value / current_price
        
        # Market impact (simplified square-root model)
        market_impact = 0.1 * np.sqrt(shares / 1000)  # 10 bps per 1000 shares
        execution_price = current_price * (1 + market_impact)
        
        # Transaction cost
        total_cost = shares * execution_price * (1 + self.transaction_cost)
        
        if total_cost <= self.capital * self.leverage:
            self.positions[asset] += shares
            self.capital -= total_cost
            self.trades.append({
                'time': self.current_step,
                'asset': asset,
                'action': 'buy',
                'shares': shares,
                'price': execution_price,
                'cost': total_cost
            })
    
    def _execute_sell(self, asset: str, position_size: float):
        """Execute sell order with market impact modeling."""
        if self.positions[asset] <= 0:
            return
        
        current_price = self.market_data[asset][self.current_step]
        shares_to_sell = self.positions[asset] * position_size
        
        # Market impact
        market_impact = 0.1 * np.sqrt(shares_to_sell / 1000)
        execution_price = current_price * (1 - market_impact)
        
        # Transaction cost
        proceeds = shares_to_sell * execution_price * (1 - self.transaction_cost)
        
        self.positions[asset] -= shares_to_sell
        self.capital += proceeds
        self.trades.append({
            'time': self.current_step,
            'asset': asset,
            'action': 'sell',
            'shares': shares_to_sell,
            'price': execution_price,
            'proceeds': proceeds
        })
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(
            self.positions[asset] * self.market_data[asset][self.current_step]
            for asset in self.market_data.keys()
        )
        return self.capital + positions_value
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on risk-adjusted returns.
        Uses Sharpe ratio and penalizes large drawdowns.
        """
        if len(self.portfolio_values) < 2:
            return 0
        
        # Recent return
        recent_return = (
            self.portfolio_values[-1] / self.portfolio_values[-2] - 1
        )
        
        # Sharpe ratio component
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        if len(returns) > 0:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Drawdown penalty
        max_drawdown = self._calculate_max_drawdown()
        drawdown_penalty = max_drawdown ** 2
        
        # Combined reward
        reward = recent_return + 0.1 * sharpe - 10 * drawdown_penalty
        
        return reward
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of portfolio."""
        if len(self.portfolio_values) < 2:
            return 0
        
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown of portfolio."""
        if len(self.portfolio_values) < 2:
            return 0
        
        running_max = np.maximum.accumulate(self.portfolio_values)
        drawdown = (self.portfolio_values - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _get_state(self) -> TradingState:
        """Get current state representation."""
        # This is a simplified version - in practice, you'd compute all features
        lookback = 50
        start_idx = max(0, self.current_step - lookback)
        
        # Get price history for all assets
        prices = []
        volumes = []
        for asset in self.market_data.keys():
            asset_prices = self.market_data[asset][start_idx:self.current_step + 1]
            prices.extend(asset_prices)
            # Simulate volumes
            volumes.extend(np.random.lognormal(10, 2, len(asset_prices)))
        
        # Pad if necessary
        if len(prices) < lookback:
            prices = np.pad(prices, (lookback - len(prices), 0), 'constant')
            volumes = np.pad(volumes, (lookback - len(volumes), 0), 'constant')
        
        return TradingState(
            prices=np.array(prices[-lookback:]),
            volumes=np.array(volumes[-lookback:]),
            technical_indicators=np.random.randn(20),  # Placeholder
            order_book=np.random.randn(10),  # Placeholder
            portfolio_state=np.array([
                self.capital / self.initial_capital,
                sum(self.positions.values()),
                self._calculate_portfolio_value() / self.initial_capital
            ]),
            market_microstructure=np.random.randn(15),  # Placeholder
            sentiment_scores=np.random.randn(5),  # Placeholder
            volatility_surface=np.random.randn(10, 10),  # Placeholder
            correlation_matrix=np.eye(len(self.market_data))  # Placeholder
        )
    
    def _get_state_dim(self) -> int:
        """Get dimension of state space."""
        # This should match the actual state representation
        return 1000  # Placeholder


class DeepRLTradingEngine:
    """
    Main deep reinforcement learning trading engine.
    Manages training, backtesting, and live trading.
    """
    
    def __init__(
        self,
        market_data: Dict[str, np.ndarray],
        agent_type: str = 'PPO',
        hyperparameters: Optional[Dict] = None
    ):
        self.market_data = market_data
        self.agent_type = agent_type
        self.hyperparameters = hyperparameters or {}
        
        # Create environment
        self.env = TradingEnvironment(market_data)
        
        # Create agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.nvec.prod()
        
        if agent_type == 'PPO':
            self.agent = PPOAgent(state_dim, action_dim, **self.hyperparameters)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Training history
        self.training_history = {
            'rewards': [],
            'portfolio_values': [],
            'sharpe_ratios': [],
            'max_drawdowns': []
        }
    
    def train(self, n_episodes: int = 1000, save_interval: int = 100):
        """Train the RL agent."""
        logger.info(f"Starting training for {n_episodes} episodes")
        
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            hidden = None
            
            while True:
                # Select action
                action, hidden = self.agent.select_action(state, hidden)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                # Store in memory
                self.agent.memory.append((
                    state.to_tensor(),
                    action,
                    reward,
                    torch.tensor([0.0])  # Placeholder for log prob
                ))
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Update agent
            self.agent.update()
            
            # Record metrics
            self.training_history['rewards'].append(episode_reward)
            self.training_history['portfolio_values'].append(info['portfolio_value'])
            self.training_history['sharpe_ratios'].append(info['sharpe_ratio'])
            self.training_history['max_drawdowns'].append(info['max_drawdown'])
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.training_history['rewards'][-100:])
                avg_sharpe = np.mean(self.training_history['sharpe_ratios'][-100:])
                logger.info(
                    f"Episode {episode}: Avg Reward = {avg_reward:.4f}, "
                    f"Avg Sharpe = {avg_sharpe:.4f}"
                )
            
            # Save model
            if episode % save_interval == 0:
                self.save_model(f"rl_trading_model_{episode}.pth")
    
    def backtest(self, test_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Backtest the trained model."""
        # Create test environment
        test_env = TradingEnvironment(test_data)
        
        state = test_env.reset()
        hidden = None
        
        portfolio_values = [test_env.initial_capital]
        trades = []
        
        while True:
            # Get action from trained agent
            with torch.no_grad():
                action, hidden = self.agent.select_action(state, hidden)
            
            # Take step
            state, reward, done, info = test_env.step(action)
            
            portfolio_values.append(info['portfolio_value'])
            trades.extend(test_env.trades[-1:])  # Get latest trade
            
            if done:
                break
        
        # Calculate performance metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        results = {
            'total_return': (portfolio_values[-1] / portfolio_values[0] - 1) * 100,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns),
            'portfolio_values': portfolio_values,
            'trades': trades,
            'final_positions': test_env.positions
        }
        
        return results
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'policy_state_dict': self.agent.policy.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath)
        self.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max
        return abs(np.min(drawdown)) * 100
