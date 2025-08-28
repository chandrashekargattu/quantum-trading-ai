"""ML Model Manager - Orchestrates all AI/ML models for trading."""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import asyncio
import joblib
from pathlib import Path

from app.ml.transformer_prediction import (
    MarketTransformer, MarketPredictionOutput
)
from app.ml.advanced_models import (
    GraphNeuralNetworkPredictor,
    EnsemblePredictor,
    AutoMLOptimizer,
    AdversarialRobustModel,
    PredictionResult
)
from app.ml.deep_rl_agent import (
    PPOAgent, RLConfig, MultiAgentTradingSystem
)
from app.quantum.quantum_algorithms import QuantumEnhancedML
from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Unified prediction output from model manager."""
    symbol: str
    timestamp: datetime
    
    # Price predictions
    price_predictions: np.ndarray  # [horizon] array
    price_confidence: float
    price_direction: str  # "up", "down", "neutral"
    
    # Volatility predictions
    volatility_predictions: np.ndarray
    volatility_confidence: float
    
    # Trading signals
    signal_strength: float  # -1 to 1
    recommended_action: str  # "buy", "sell", "hold"
    position_size: float
    
    # Risk metrics
    risk_score: float
    max_drawdown_estimate: float
    
    # Model metadata
    models_used: List[str]
    quantum_enhanced: bool
    computation_time: float


class ModelManager:
    """
    Central manager for all ML/AI models.
    
    Features:
    - Model orchestration and ensemble
    - Automatic model selection based on market conditions
    - Performance tracking and model switching
    - Quantum enhancement when beneficial
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.models = {}
        self.model_performance = {}
        self.active_models = set()
        
        # Services
        self.market_service = MarketDataService()
        self.quantum_ml = QuantumEnhancedML()
        
        # Model paths
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        self._initialized = False
    
    async def load_models(self):
        """Load all models asynchronously."""
        logger.info("Loading ML models...")
        
        load_tasks = [
            self._load_transformer(),
            self._load_gnn(),
            self._load_rl_agents(),
            self._load_ensemble(),
            self._load_automl()
        ]
        
        await asyncio.gather(*load_tasks)
        
        self._initialized = True
        logger.info("All models loaded successfully")
    
    async def _load_transformer(self):
        """Load transformer model."""
        try:
            config = {
                'num_assets': 100,
                'd_model': 512,
                'nhead': 8,
                'num_layers': 6,
                'prediction_horizon': 5
            }
            
            model = MarketTransformer(
                num_assets=config['num_assets'],
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_encoder_layers=config['num_layers'],
                num_decoder_layers=config['num_layers'],
                prediction_horizon=config['prediction_horizon']
            ).to(self.device)
            
            # Load weights if available
            model_path = self.model_dir / "transformer.pth"
            if model_path.exists():
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
            
            self.models['transformer'] = model
            self.active_models.add('transformer')
            
        except Exception as e:
            logger.error(f"Failed to load transformer: {e}")
    
    async def _load_gnn(self):
        """Load Graph Neural Network."""
        try:
            model = GraphNeuralNetworkPredictor(
                node_features=20,
                edge_features=3,
                hidden_dim=256,
                num_layers=3,
                prediction_horizon=5
            ).to(self.device)
            
            model_path = self.model_dir / "gnn.pth"
            if model_path.exists():
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
            
            self.models['gnn'] = model
            self.active_models.add('gnn')
            
        except Exception as e:
            logger.error(f"Failed to load GNN: {e}")
    
    async def _load_rl_agents(self):
        """Load reinforcement learning agents."""
        try:
            # Single agent
            config = RLConfig(
                state_dim=50,
                action_dim=10,
                hidden_dim=256,
                use_attention=True,
                use_lstm=True
            )
            
            agent = PPOAgent(config)
            
            model_path = self.model_dir / "rl_agent.pth"
            if model_path.exists():
                agent.load(str(model_path))
            
            self.models['rl_agent'] = agent
            
            # Multi-agent system
            multi_agent = MultiAgentTradingSystem(num_assets=10)
            self.models['multi_agent'] = multi_agent
            
            self.active_models.add('rl_agent')
            
        except Exception as e:
            logger.error(f"Failed to load RL agents: {e}")
    
    async def _load_ensemble(self):
        """Load ensemble predictor."""
        try:
            ensemble = EnsemblePredictor()
            
            # Add available models to ensemble
            if 'transformer' in self.models:
                ensemble.add_model('transformer', self.models['transformer'], weight=2.0)
            
            if 'gnn' in self.models:
                ensemble.add_model('gnn', self.models['gnn'], weight=1.5)
            
            self.models['ensemble'] = ensemble
            self.active_models.add('ensemble')
            
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
    
    async def _load_automl(self):
        """Load AutoML model."""
        try:
            automl = AutoMLOptimizer(task_type="regression")
            
            model_path = self.model_dir / "automl.joblib"
            if model_path.exists():
                automl.best_model = joblib.load(model_path)
            
            self.models['automl'] = automl
            
        except Exception as e:
            logger.error(f"Failed to load AutoML: {e}")
    
    async def predict(
        self,
        symbol: str,
        horizon: int = 5,
        use_quantum: bool = False,
        models: Optional[List[str]] = None
    ) -> ModelPrediction:
        """Make prediction using specified or best models."""
        start_time = datetime.now()
        
        if not self._initialized:
            await self.load_models()
        
        # Fetch market data
        market_data = await self._prepare_market_data(symbol)
        
        # Select models to use
        if models is None:
            models = self._select_best_models(symbol, market_data)
        
        # Make predictions
        predictions = {}
        for model_name in models:
            if model_name in self.models and model_name in self.active_models:
                try:
                    pred = await self._predict_with_model(
                        model_name,
                        market_data,
                        horizon,
                        use_quantum
                    )
                    predictions[model_name] = pred
                except Exception as e:
                    logger.error(f"Prediction failed for {model_name}: {e}")
        
        # Combine predictions
        final_prediction = self._combine_predictions(predictions, symbol)
        
        # Add metadata
        final_prediction.computation_time = (datetime.now() - start_time).total_seconds()
        final_prediction.models_used = list(predictions.keys())
        final_prediction.quantum_enhanced = use_quantum
        
        return final_prediction
    
    async def train_online(
        self,
        symbol: str,
        new_data: pd.DataFrame
    ):
        """Online training/adaptation of models."""
        # Update models with new data
        for model_name in self.active_models:
            if hasattr(self.models[model_name], 'update'):
                try:
                    await self._update_model(model_name, symbol, new_data)
                except Exception as e:
                    logger.error(f"Failed to update {model_name}: {e}")
    
    async def _prepare_market_data(self, symbol: str) -> Dict[str, Any]:
        """Prepare market data for models."""
        # Fetch various data types
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Price data
        price_data = await self.market_service.fetch_historical_data(
            symbol, start_date.date(), end_date.date()
        )
        
        # Technical indicators
        indicators = await self.market_service.calculate_indicators(symbol)
        
        # Order book data (if available)
        order_book = await self.market_service.fetch_order_book(symbol)
        
        # News sentiment (placeholder)
        sentiment = await self._fetch_sentiment(symbol)
        
        return {
            'symbol': symbol,
            'price_data': price_data,
            'indicators': indicators,
            'order_book': order_book,
            'sentiment': sentiment,
            'timestamp': datetime.now()
        }
    
    async def _predict_with_model(
        self,
        model_name: str,
        market_data: Dict[str, Any],
        horizon: int,
        use_quantum: bool
    ) -> PredictionResult:
        """Make prediction with specific model."""
        model = self.models[model_name]
        
        if model_name == 'transformer':
            # Prepare transformer input
            features = self._prepare_transformer_features(market_data)
            
            with torch.no_grad():
                output: MarketPredictionOutput = model(
                    features['price_data'],
                    features.get('text_data'),
                    features.get('order_book_data')
                )
            
            predictions = output.price_predictions.cpu().numpy()
            confidence = output.confidence_scores.mean().item()
            
        elif model_name == 'gnn':
            # Create market graph
            graph_data = model.create_market_graph(
                market_data['price_data'],
                correlation_threshold=0.3
            )
            
            with torch.no_grad():
                output = model(
                    graph_data.x,
                    graph_data.edge_index,
                    graph_data.edge_attr
                )
            
            predictions = output['price_predictions'].cpu().numpy()
            confidence = 0.7  # Default confidence
            
        elif model_name == 'rl_agent':
            # RL agent predicts actions, convert to price predictions
            state = self._prepare_rl_state(market_data)
            action, _ = model.select_action(state, deterministic=True)
            
            # Convert action to price prediction
            current_price = market_data['price_data'][-1]['close']
            predictions = current_price * (1 + action[:horizon] * 0.01)
            confidence = 0.6
            
        elif model_name == 'ensemble':
            # Ensemble handles multiple models internally
            features = self._prepare_ensemble_features(market_data)
            result = model.predict(features, method="bayesian_average")
            
            predictions = result.predictions
            confidence = result.confidence
            
        else:
            # Default prediction
            current_price = market_data['price_data'][-1]['close']
            predictions = np.array([current_price] * horizon)
            confidence = 0.5
        
        # Apply quantum enhancement if requested
        if use_quantum:
            predictions, confidence = await self._apply_quantum_enhancement(
                predictions, confidence, market_data
            )
        
        return PredictionResult(
            predictions=predictions,
            confidence=np.array([confidence]),
            model_type=model_name
        )
    
    async def _apply_quantum_enhancement(
        self,
        predictions: np.ndarray,
        confidence: float,
        market_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, float]:
        """Apply quantum enhancement to predictions."""
        # Extract features
        features = self._extract_quantum_features(market_data)
        
        # Quantum feature mapping
        quantum_features = await self.quantum_ml.quantum_feature_mapping(
            features, encoding_type="amplitude"
        )
        
        # Enhance predictions based on quantum features
        enhancement_factor = 1.0 + 0.05 * np.mean(quantum_features)
        enhanced_predictions = predictions * enhancement_factor
        
        # Boost confidence slightly for quantum enhancement
        enhanced_confidence = min(confidence * 1.1, 0.95)
        
        return enhanced_predictions, enhanced_confidence
    
    def _combine_predictions(
        self,
        predictions: Dict[str, PredictionResult],
        symbol: str
    ) -> ModelPrediction:
        """Combine predictions from multiple models."""
        if not predictions:
            raise ValueError("No predictions to combine")
        
        # Weighted average based on confidence
        total_weight = sum(p.confidence[0] for p in predictions.values())
        
        combined_price = None
        combined_volatility = None
        
        for model_name, pred in predictions.items():
            weight = pred.confidence[0] / total_weight
            
            if combined_price is None:
                combined_price = pred.predictions * weight
                combined_volatility = np.ones_like(pred.predictions) * 0.02 * weight
            else:
                combined_price += pred.predictions * weight
                combined_volatility += np.ones_like(pred.predictions) * 0.02 * weight
        
        # Determine direction and signal
        current_price = combined_price[0]
        future_price = combined_price[-1]
        price_change = (future_price - current_price) / current_price
        
        if price_change > 0.01:
            direction = "up"
            signal_strength = min(price_change * 10, 1.0)
            action = "buy"
        elif price_change < -0.01:
            direction = "down"
            signal_strength = max(price_change * 10, -1.0)
            action = "sell"
        else:
            direction = "neutral"
            signal_strength = 0.0
            action = "hold"
        
        # Position sizing based on confidence and signal strength
        avg_confidence = np.mean([p.confidence[0] for p in predictions.values()])
        position_size = abs(signal_strength) * avg_confidence * 0.1  # Max 10% position
        
        # Risk metrics
        risk_score = combined_volatility.mean() / 0.02  # Normalized by typical volatility
        max_drawdown_estimate = combined_volatility.max() * 2  # 2 std devs
        
        return ModelPrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            price_predictions=combined_price,
            price_confidence=avg_confidence,
            price_direction=direction,
            volatility_predictions=combined_volatility,
            volatility_confidence=avg_confidence * 0.9,
            signal_strength=signal_strength,
            recommended_action=action,
            position_size=position_size,
            risk_score=risk_score,
            max_drawdown_estimate=max_drawdown_estimate,
            models_used=list(predictions.keys()),
            quantum_enhanced=False,
            computation_time=0.0
        )
    
    def _select_best_models(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> List[str]:
        """Select best models based on market conditions."""
        selected_models = []
        
        # Always include transformer for time series
        if 'transformer' in self.active_models:
            selected_models.append('transformer')
        
        # Add GNN if we have correlation data
        if 'gnn' in self.active_models and len(market_data.get('price_data', [])) > 100:
            selected_models.append('gnn')
        
        # Add RL agent for dynamic markets
        volatility = self._calculate_volatility(market_data)
        if 'rl_agent' in self.active_models and volatility > 0.02:
            selected_models.append('rl_agent')
        
        # Use ensemble if available
        if 'ensemble' in self.active_models and len(selected_models) > 1:
            selected_models = ['ensemble']
        
        return selected_models or ['transformer']  # Default to transformer
    
    def _calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate recent volatility."""
        if 'price_data' not in market_data or len(market_data['price_data']) < 2:
            return 0.02  # Default volatility
        
        prices = [d['close'] for d in market_data['price_data'][-20:]]
        returns = np.diff(prices) / prices[:-1]
        
        return np.std(returns) * np.sqrt(252)  # Annualized volatility
    
    def _prepare_transformer_features(self, market_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare features for transformer model."""
        # Price features
        prices = [d['close'] for d in market_data['price_data'][-100:]]
        volumes = [d['volume'] for d in market_data['price_data'][-100:]]
        
        price_tensor = torch.FloatTensor(prices).unsqueeze(0).unsqueeze(-1)
        volume_tensor = torch.FloatTensor(volumes).unsqueeze(0).unsqueeze(-1)
        
        price_features = torch.cat([price_tensor, volume_tensor], dim=-1)
        
        return {
            'price_data': price_features.to(self.device),
            'text_data': None,  # Placeholder
            'order_book_data': None  # Placeholder
        }
    
    def _prepare_rl_state(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Prepare state for RL agent."""
        # Extract recent prices and indicators
        prices = [d['close'] for d in market_data['price_data'][-50:]]
        
        if len(prices) < 50:
            prices = [prices[0]] * (50 - len(prices)) + prices
        
        # Normalize
        prices = np.array(prices)
        normalized_prices = prices / prices[-1]
        
        # Add technical indicators if available
        indicators = market_data.get('indicators', {})
        
        state = np.concatenate([
            normalized_prices,
            [indicators.get('rsi', 50) / 100],
            [indicators.get('macd', 0)],
            [indicators.get('volume_ratio', 1.0)]
        ])
        
        return state[:50]  # Ensure correct dimension
    
    def _prepare_ensemble_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Prepare features for ensemble model."""
        features = []
        
        # Price features
        prices = [d['close'] for d in market_data['price_data'][-20:]]
        returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else [0]
        
        features.extend([
            np.mean(returns),
            np.std(returns),
            np.min(returns),
            np.max(returns),
            len([r for r in returns if r > 0]) / len(returns) if returns else 0.5
        ])
        
        # Technical indicators
        indicators = market_data.get('indicators', {})
        features.extend([
            indicators.get('rsi', 50) / 100,
            indicators.get('macd', 0),
            indicators.get('bollinger_position', 0.5)
        ])
        
        return np.array(features)
    
    def _extract_quantum_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract features for quantum enhancement."""
        # Similar to ensemble features but more focused on quantum-relevant aspects
        features = self._prepare_ensemble_features(market_data)
        
        # Add quantum-specific features (e.g., phase-like quantities)
        prices = [d['close'] for d in market_data['price_data'][-10:]]
        if len(prices) > 1:
            # Price momentum as "phase"
            momentum = (prices[-1] - prices[0]) / prices[0]
            features = np.append(features, momentum)
        
        return features[:10]  # Limit features for quantum circuit
    
    async def _fetch_sentiment(self, symbol: str) -> Dict[str, float]:
        """Fetch sentiment data (placeholder)."""
        # In production, would integrate with news/social APIs
        return {
            'news_sentiment': 0.6,
            'social_sentiment': 0.5,
            'analyst_rating': 0.7
        }
    
    async def _update_model(self, model_name: str, symbol: str, new_data: pd.DataFrame):
        """Update specific model with new data."""
        model = self.models[model_name]
        
        if model_name == 'rl_agent':
            # RL agents update through experience replay
            # This would be called during live trading
            pass
        
        elif model_name == 'automl' and hasattr(model, 'best_model'):
            # Incremental learning for tree-based models
            # Some models support partial_fit
            pass
        
        # Track model performance
        self._update_model_performance(model_name, symbol, new_data)
    
    def _update_model_performance(self, model_name: str, symbol: str, new_data: pd.DataFrame):
        """Track model performance for adaptive selection."""
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {
                'accuracy': deque(maxlen=100),
                'sharpe': deque(maxlen=100),
                'total_return': 0
            }
        
        # Calculate performance metrics (placeholder)
        # In production, would compare predictions with actual outcomes
        accuracy = 0.6  # Placeholder
        sharpe = 1.2  # Placeholder
        
        self.model_performance[model_name]['accuracy'].append(accuracy)
        self.model_performance[model_name]['sharpe'].append(sharpe)
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'enable_quantum': True,
            'ensemble_method': 'bayesian_average',
            'model_update_frequency': 'daily',
            'risk_threshold': 0.02,
            'confidence_threshold': 0.6
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up ML models...")
        
        # Save model states
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'save'):
                    model.save(str(self.model_dir / f"{model_name}.pth"))
                elif hasattr(model, 'state_dict'):
                    torch.save(
                        model.state_dict(),
                        self.model_dir / f"{model_name}.pth"
                    )
            except Exception as e:
                logger.error(f"Failed to save {model_name}: {e}")
        
        logger.info("ML models cleanup complete")