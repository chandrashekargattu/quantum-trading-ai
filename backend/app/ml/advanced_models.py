"""Advanced ML models for trading including GNNs, RL, and ensemble methods."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from collections import deque
import gym
from gym import spaces
import stable_baselines3
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import optuna
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Unified prediction result from various models."""
    predictions: np.ndarray
    confidence: np.ndarray
    feature_importance: Optional[Dict[str, float]] = None
    model_type: str = ""
    metadata: Dict[str, Any] = None


class GraphNeuralNetworkPredictor(nn.Module):
    """
    Graph Neural Network for capturing market structure and relationships.
    
    Models the market as a graph where:
    - Nodes: Assets/stocks
    - Edges: Correlations, sector relationships, supply chain connections
    """
    
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        prediction_horizon: int = 5
    ):
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon
        
        # Node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=8, concat=True)
                )
            else:
                self.gnn_layers.append(
                    GATConv(hidden_dim * 8, hidden_dim, heads=8, concat=True)
                )
        
        # Edge attention
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim * 8, num_heads=8, dropout=dropout
        )
        
        # Output layers
        self.price_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prediction_horizon)
        )
        
        self.volatility_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prediction_horizon)
        )
        
        self.correlation_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GNN.
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            batch: Batch assignment for nodes
        """
        # Encode nodes
        x = self.node_encoder(x)
        
        # Apply edge attention if edge features provided
        edge_weight = None
        if edge_attr is not None:
            edge_weight = self.edge_mlp(edge_attr).squeeze()
            edge_weight = torch.sigmoid(edge_weight)
        
        # Graph convolutions
        for i, layer in enumerate(self.gnn_layers):
            x_residual = x
            x = layer(x, edge_index, edge_weight)
            x = F.relu(x)
            
            # Skip connection after first layer
            if i > 0 and x.shape == x_residual.shape:
                x = x + x_residual
        
        # Global pooling
        if batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_global = torch.cat([x_mean, x_max], dim=1)
        else:
            x_global = x
        
        # Predictions
        price_pred = self.price_predictor(x_global)
        vol_pred = self.volatility_predictor(x_global)
        
        # Pairwise correlation predictions
        if batch is None:
            # Get pairwise node representations
            num_nodes = x.shape[0]
            x1 = x.unsqueeze(1).expand(-1, num_nodes, -1)
            x2 = x.unsqueeze(0).expand(num_nodes, -1, -1)
            x_pairs = torch.cat([x1, x2], dim=-1)
            corr_pred = self.correlation_predictor(x_pairs.view(-1, x_pairs.shape[-1]))
            corr_pred = corr_pred.view(num_nodes, num_nodes)
        else:
            corr_pred = None
        
        return {
            'price_predictions': price_pred,
            'volatility_predictions': vol_pred,
            'correlation_predictions': corr_pred,
            'node_embeddings': x,
            'global_embedding': x_global if batch is not None else None
        }
    
    def create_market_graph(
        self,
        returns_data: pd.DataFrame,
        sector_data: Optional[Dict[str, str]] = None,
        correlation_threshold: float = 0.3
    ) -> Data:
        """Create graph representation of market."""
        symbols = returns_data.columns.tolist()
        n_symbols = len(symbols)
        
        # Node features: returns statistics
        node_features = []
        for symbol in symbols:
            features = [
                returns_data[symbol].mean(),
                returns_data[symbol].std(),
                returns_data[symbol].skew(),
                returns_data[symbol].kurtosis(),
                returns_data[symbol].min(),
                returns_data[symbol].max()
            ]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge construction based on correlation
        corr_matrix = returns_data.corr()
        edge_index = []
        edge_attr = []
        
        for i in range(n_symbols):
            for j in range(i + 1, n_symbols):
                corr = corr_matrix.iloc[i, j]
                
                if abs(corr) > correlation_threshold:
                    # Add bidirectional edges
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    
                    # Edge features
                    edge_features = [
                        corr,
                        abs(corr),
                        1.0 if sector_data and sector_data.get(symbols[i]) == sector_data.get(symbols[j]) else 0.0
                    ]
                    edge_attr.append(edge_features)
                    edge_attr.append(edge_features)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class TradingEnvironment(gym.Env):
    """
    OpenAI Gym environment for reinforcement learning trading.
    
    Features:
    - Continuous action space for portfolio allocation
    - Rich observation space including price, volume, indicators
    - Realistic transaction costs and slippage
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000,
        transaction_cost: float = 0.001,
        max_position_size: float = 0.2,
        lookback_window: int = 50
    ):
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        
        # State tracking
        self.current_step = 0
        self.balance = initial_balance
        self.positions = {}
        self.portfolio_value_history = []
        
        # Action space: portfolio weights for each asset
        n_assets = len(data.columns)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(n_assets,), dtype=np.float32
        )
        
        # Observation space
        obs_dim = n_assets * (6 + lookback_window)  # Price features + history
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.positions = {symbol: 0 for symbol in self.data.columns}
        self.portfolio_value_history = []
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        # Normalize actions to valid portfolio weights
        action = np.clip(action, -self.max_position_size, self.max_position_size)
        action = action / np.sum(np.abs(action)) if np.sum(np.abs(action)) > 0 else action
        
        # Get current prices
        current_prices = self.data.iloc[self.current_step]
        prev_prices = self.data.iloc[self.current_step - 1]
        
        # Calculate returns
        returns = (current_prices - prev_prices) / prev_prices
        
        # Update positions
        target_positions = {}
        total_value = self._get_portfolio_value()
        
        for i, symbol in enumerate(self.data.columns):
            target_value = total_value * action[i]
            target_shares = int(target_value / current_prices[symbol])
            target_positions[symbol] = target_shares
        
        # Calculate transaction costs
        transaction_cost = 0
        for symbol in self.data.columns:
            shares_traded = abs(target_positions[symbol] - self.positions.get(symbol, 0))
            transaction_cost += shares_traded * current_prices[symbol] * self.transaction_cost
        
        # Update positions and balance
        for symbol in self.data.columns:
            share_diff = target_positions[symbol] - self.positions.get(symbol, 0)
            self.balance -= share_diff * current_prices[symbol]
        
        self.positions = target_positions
        self.balance -= transaction_cost
        
        # Calculate reward
        new_portfolio_value = self._get_portfolio_value()
        self.portfolio_value_history.append(new_portfolio_value)
        
        if len(self.portfolio_value_history) > 1:
            portfolio_return = (new_portfolio_value - self.portfolio_value_history[-2]) / self.portfolio_value_history[-2]
            
            # Sharpe-based reward
            if len(self.portfolio_value_history) > 20:
                recent_returns = np.diff(self.portfolio_value_history[-21:]) / self.portfolio_value_history[-21:-1]
                sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8) * np.sqrt(252)
                reward = sharpe
            else:
                reward = portfolio_return
        else:
            reward = 0
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        # Get next observation
        obs = self._get_observation()
        
        info = {
            'portfolio_value': new_portfolio_value,
            'positions': self.positions.copy(),
            'balance': self.balance,
            'transaction_cost': transaction_cost
        }
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = []
        
        # Get historical data
        start_idx = max(0, self.current_step - self.lookback_window)
        historical_data = self.data.iloc[start_idx:self.current_step + 1]
        
        for symbol in self.data.columns:
            # Price features
            prices = historical_data[symbol].values
            returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([0])
            
            features = [
                prices[-1] / prices[0] if len(prices) > 0 and prices[0] != 0 else 1,  # Normalized price
                np.mean(returns) if len(returns) > 0 else 0,
                np.std(returns) if len(returns) > 0 else 0,
                np.min(returns) if len(returns) > 0 else 0,
                np.max(returns) if len(returns) > 0 else 0,
                self.positions.get(symbol, 0) / 1000  # Normalized position
            ]
            
            # Add historical prices
            if len(prices) < self.lookback_window:
                prices = np.pad(prices, (self.lookback_window - len(prices), 0), 'constant')
            
            obs.extend(features)
            obs.extend(prices)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        value = self.balance
        current_prices = self.data.iloc[self.current_step]
        
        for symbol, shares in self.positions.items():
            value += shares * current_prices[symbol]
        
        return value


class AdvancedRLAgent:
    """
    Advanced reinforcement learning agent with multiple algorithms.
    
    Supports:
    - PPO (Proximal Policy Optimization)
    - SAC (Soft Actor-Critic)
    - TD3 (Twin Delayed DDPG)
    - A2C (Advantage Actor-Critic)
    """
    
    def __init__(
        self,
        algorithm: str = "PPO",
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_steps: int = 2048,
        policy_kwargs: Optional[Dict] = None
    ):
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_steps = n_steps
        
        # Default policy network architecture
        if policy_kwargs is None:
            policy_kwargs = {
                "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
                "activation_fn": nn.ReLU
            }
        self.policy_kwargs = policy_kwargs
        
        self.model = None
        self.env = None
    
    def create_model(self, env: gym.Env):
        """Create RL model based on specified algorithm."""
        self.env = DummyVecEnv([lambda: env])
        
        if self.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                policy_kwargs=self.policy_kwargs,
                verbose=1
            )
        elif self.algorithm == "SAC":
            self.model = SAC(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                policy_kwargs=self.policy_kwargs,
                verbose=1
            )
        elif self.algorithm == "TD3":
            self.model = TD3(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                policy_kwargs=self.policy_kwargs,
                verbose=1
            )
        elif self.algorithm == "A2C":
            self.model = A2C(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                policy_kwargs=self.policy_kwargs,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000
    ):
        """Train the RL agent."""
        callbacks = []
        
        if eval_env is not None:
            eval_callback = EvalCallback(
                DummyVecEnv([lambda: eval_env]),
                best_model_save_path="./rl_models/",
                log_path="./rl_logs/",
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )
    
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Make prediction using trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def save(self, path: str):
        """Save trained model."""
        if self.model is not None:
            self.model.save(path)
    
    def load(self, path: str, env: gym.Env):
        """Load trained model."""
        self.env = DummyVecEnv([lambda: env])
        
        if self.algorithm == "PPO":
            self.model = PPO.load(path, env=self.env)
        elif self.algorithm == "SAC":
            self.model = SAC.load(path, env=self.env)
        elif self.algorithm == "TD3":
            self.model = TD3.load(path, env=self.env)
        elif self.algorithm == "A2C":
            self.model = A2C.load(path, env=self.env)


class AdversarialRobustModel:
    """
    Adversarial training for robust predictions.
    
    Makes models resistant to:
    - Market manipulation
    - Data poisoning
    - Adversarial examples
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        epsilon: float = 0.01,
        alpha: float = 0.001,
        num_iter: int = 10
    ):
        self.base_model = base_model
        self.epsilon = epsilon  # Maximum perturbation
        self.alpha = alpha  # Step size
        self.num_iter = num_iter  # PGD iterations
    
    def fgsm_attack(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        loss_fn: nn.Module
    ) -> torch.Tensor:
        """Fast Gradient Sign Method attack."""
        data.requires_grad = True
        
        output = self.base_model(data)
        loss = loss_fn(output, target)
        
        self.base_model.zero_grad()
        loss.backward()
        
        data_grad = data.grad.data
        sign_data_grad = data_grad.sign()
        
        perturbed_data = data + self.epsilon * sign_data_grad
        return perturbed_data
    
    def pgd_attack(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        loss_fn: nn.Module
    ) -> torch.Tensor:
        """Projected Gradient Descent attack."""
        perturbed_data = data.clone().detach()
        perturbed_data.requires_grad = True
        
        for i in range(self.num_iter):
            output = self.base_model(perturbed_data)
            loss = loss_fn(output, target)
            
            self.base_model.zero_grad()
            loss.backward()
            
            data_grad = perturbed_data.grad.data
            perturbed_data = perturbed_data + self.alpha * data_grad.sign()
            
            # Project back to epsilon ball
            perturbation = torch.clamp(
                perturbed_data - data,
                min=-self.epsilon,
                max=self.epsilon
            )
            perturbed_data = data + perturbation
            perturbed_data = perturbed_data.detach()
            perturbed_data.requires_grad = True
        
        return perturbed_data
    
    def adversarial_training_step(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        attack_ratio: float = 0.5
    ) -> float:
        """Single adversarial training step."""
        batch_size = data.shape[0]
        num_adversarial = int(batch_size * attack_ratio)
        
        # Split batch
        clean_data = data[num_adversarial:]
        adv_data = data[:num_adversarial]
        clean_target = target[num_adversarial:]
        adv_target = target[:num_adversarial]
        
        # Generate adversarial examples
        if num_adversarial > 0:
            adv_data = self.pgd_attack(adv_data, adv_target, loss_fn)
        
        # Combine clean and adversarial
        combined_data = torch.cat([adv_data, clean_data], dim=0)
        combined_target = torch.cat([adv_target, clean_target], dim=0)
        
        # Forward pass
        output = self.base_model(combined_data)
        loss = loss_fn(output, combined_target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


class EnsemblePredictor:
    """
    Advanced ensemble methods combining multiple models.
    
    Includes:
    - Stacking
    - Blending
    - Bayesian model averaging
    - Dynamic weighting based on recent performance
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.performance_history = defaultdict(list)
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add model to ensemble."""
        self.models[name] = model
        self.weights[name] = weight
    
    def predict(
        self,
        X: np.ndarray,
        method: str = "weighted_average"
    ) -> PredictionResult:
        """Make ensemble prediction."""
        predictions = {}
        confidences = {}
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                    conf = np.max(pred, axis=1)
                elif hasattr(model, 'predict'):
                    pred = model.predict(X)
                    conf = np.ones_like(pred) * 0.5  # Default confidence
                else:
                    continue
                
                predictions[name] = pred
                confidences[name] = conf
                
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models made successful predictions")
        
        # Combine predictions based on method
        if method == "weighted_average":
            final_pred = self._weighted_average(predictions)
        elif method == "stacking":
            final_pred = self._stacking(predictions, X)
        elif method == "bayesian_average":
            final_pred = self._bayesian_average(predictions, confidences)
        elif method == "dynamic":
            final_pred = self._dynamic_weighting(predictions)
        else:
            final_pred = self._weighted_average(predictions)
        
        # Calculate ensemble confidence
        ensemble_confidence = np.mean([
            conf * self.weights.get(name, 1.0)
            for name, conf in confidences.items()
        ])
        
        return PredictionResult(
            predictions=final_pred,
            confidence=ensemble_confidence,
            model_type="ensemble",
            metadata={
                "method": method,
                "models": list(self.models.keys()),
                "weights": self.weights
            }
        )
    
    def _weighted_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple weighted average of predictions."""
        total_weight = sum(self.weights.get(name, 1.0) for name in predictions)
        
        weighted_sum = None
        for name, pred in predictions.items():
            weight = self.weights.get(name, 1.0) / total_weight
            
            if weighted_sum is None:
                weighted_sum = pred * weight
            else:
                weighted_sum += pred * weight
        
        return weighted_sum
    
    def _stacking(
        self,
        predictions: Dict[str, np.ndarray],
        original_features: np.ndarray
    ) -> np.ndarray:
        """Stacking ensemble with meta-learner."""
        # Combine predictions as features
        stacked_features = np.column_stack(list(predictions.values()))
        
        # Add original features (optional)
        if original_features.shape[0] == stacked_features.shape[0]:
            combined_features = np.hstack([stacked_features, original_features])
        else:
            combined_features = stacked_features
        
        # Use simple meta-learner (could be trained separately)
        # For now, use weighted combination based on feature importance
        weights = np.array([self.weights.get(name, 1.0) for name in predictions])
        weights = weights / weights.sum()
        
        return stacked_features @ weights
    
    def _bayesian_average(
        self,
        predictions: Dict[str, np.ndarray],
        confidences: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Bayesian model averaging."""
        # Calculate posterior weights based on confidence
        posterior_weights = {}
        
        for name in predictions:
            prior = self.weights.get(name, 1.0)
            likelihood = np.mean(confidences[name])
            posterior_weights[name] = prior * likelihood
        
        # Normalize
        total = sum(posterior_weights.values())
        posterior_weights = {k: v/total for k, v in posterior_weights.items()}
        
        # Weighted combination
        result = None
        for name, pred in predictions.items():
            weight = posterior_weights[name]
            
            if result is None:
                result = pred * weight
            else:
                result += pred * weight
        
        return result
    
    def _dynamic_weighting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Dynamic weighting based on recent performance."""
        # Use recent performance to adjust weights
        dynamic_weights = {}
        
        for name in predictions:
            if name in self.performance_history and len(self.performance_history[name]) > 0:
                # Use exponential moving average of recent performance
                recent_perf = self.performance_history[name][-10:]
                ema_weight = 2 / (len(recent_perf) + 1)
                ema = sum(p * (1 - ema_weight) ** i for i, p in enumerate(reversed(recent_perf)))
                dynamic_weights[name] = ema
            else:
                dynamic_weights[name] = self.weights.get(name, 1.0)
        
        # Normalize
        total = sum(dynamic_weights.values())
        dynamic_weights = {k: v/total for k, v in dynamic_weights.items()}
        
        # Apply weights
        result = None
        for name, pred in predictions.items():
            weight = dynamic_weights[name]
            
            if result is None:
                result = pred * weight
            else:
                result += pred * weight
        
        return result
    
    def update_performance(self, model_name: str, performance_score: float):
        """Update model performance history."""
        self.performance_history[model_name].append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history[model_name]) > 100:
            self.performance_history[model_name].pop(0)


class AutoMLOptimizer:
    """
    Automated machine learning optimization using Optuna.
    
    Features:
    - Hyperparameter optimization
    - Model selection
    - Feature engineering automation
    - Neural architecture search
    """
    
    def __init__(self, task_type: str = "regression"):
        self.task_type = task_type
        self.study = None
        self.best_model = None
        self.best_params = None
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100,
        models_to_try: List[str] = None
    ):
        """Run AutoML optimization."""
        if models_to_try is None:
            models_to_try = ["xgboost", "lightgbm", "catboost", "random_forest"]
        
        def objective(trial):
            model_name = trial.suggest_categorical("model", models_to_try)
            
            if model_name == "xgboost":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = xgb.XGBRegressor(**params, random_state=42)
                
            elif model_name == "lightgbm":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                }
                model = lgb.LGBMRegressor(**params, random_state=42)
                
            elif model_name == "catboost":
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 500),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                }
                model = cb.CatBoostRegressor(**params, random_state=42, verbose=False)
                
            else:  # random_forest
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                }
                model = RandomForestRegressor(**params, random_state=42)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            predictions = model.predict(X_val)
            
            if self.task_type == "regression":
                score = -np.mean((predictions - y_val) ** 2)  # Negative MSE
            else:
                score = np.mean(predictions == y_val)  # Accuracy
            
            return score
        
        # Create study
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        self.best_params = self.study.best_params
        
        # Train best model
        model_name = self.best_params['model']
        
        if model_name == "xgboost":
            model_params = {k: v for k, v in self.best_params.items() if k != 'model'}
            self.best_model = xgb.XGBRegressor(**model_params, random_state=42)
        elif model_name == "lightgbm":
            model_params = {k: v for k, v in self.best_params.items() if k != 'model'}
            self.best_model = lgb.LGBMRegressor(**model_params, random_state=42)
        elif model_name == "catboost":
            model_params = {k: v for k, v in self.best_params.items() if k != 'model'}
            self.best_model = cb.CatBoostRegressor(**model_params, random_state=42, verbose=False)
        else:
            model_params = {k: v for k, v in self.best_params.items() if k != 'model'}
            self.best_model = RandomForestRegressor(**model_params, random_state=42)
        
        # Train on full training set
        self.best_model.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using best model."""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        return self.best_model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from best model."""
        if self.best_model is None:
            return {}
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}
        
        return {}
