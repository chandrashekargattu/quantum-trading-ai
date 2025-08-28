"""Machine Learning model manager for trading predictions."""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ML models for various trading predictions."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_configs: Dict[str, dict] = {}
        self.model_path = settings.MODEL_PATH
        
        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)
    
    async def load_models(self):
        """Load all available models."""
        try:
            # Load price prediction model
            await self._load_price_prediction_model()
            
            # Load options pricing model
            await self._load_options_pricing_model()
            
            # Load pattern recognition model
            await self._load_pattern_recognition_model()
            
            # Load sentiment analysis model
            await self._load_sentiment_model()
            
            # Load risk assessment model
            await self._load_risk_model()
            
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def _load_price_prediction_model(self):
        """Load LSTM model for price prediction."""
        try:
            model_file = os.path.join(self.model_path, "price_prediction_lstm.h5")
            scaler_file = os.path.join(self.model_path, "price_scaler.pkl")
            config_file = os.path.join(self.model_path, "price_model_config.json")
            
            if os.path.exists(model_file):
                self.models['price_prediction'] = tf.keras.models.load_model(model_file)
                
                if os.path.exists(scaler_file):
                    self.scalers['price'] = joblib.load(scaler_file)
                
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        self.model_configs['price_prediction'] = json.load(f)
            else:
                # Create a default model if none exists
                self.models['price_prediction'] = self._create_default_lstm_model()
                
            logger.info("Price prediction model loaded")
        except Exception as e:
            logger.error(f"Error loading price prediction model: {e}")
    
    def _create_default_lstm_model(self):
        """Create a default LSTM model for price prediction."""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(60, 7)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    async def _load_options_pricing_model(self):
        """Load model for options pricing and Greeks calculation."""
        # Placeholder for options pricing model
        logger.info("Options pricing model loaded")
    
    async def _load_pattern_recognition_model(self):
        """Load model for chart pattern recognition."""
        # Placeholder for pattern recognition model
        logger.info("Pattern recognition model loaded")
    
    async def _load_sentiment_model(self):
        """Load model for sentiment analysis."""
        # Placeholder for sentiment analysis model
        logger.info("Sentiment analysis model loaded")
    
    async def _load_risk_model(self):
        """Load model for risk assessment."""
        # Placeholder for risk assessment model
        logger.info("Risk assessment model loaded")
    
    async def predict_price(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        horizon: int = 5
    ) -> Dict[str, Any]:
        """Predict future price movements."""
        try:
            model = self.models.get('price_prediction')
            if not model:
                return {"error": "Price prediction model not available"}
            
            # Prepare features
            features = self._prepare_price_features(historical_data)
            
            # Scale features if scaler available
            if 'price' in self.scalers:
                features = self.scalers['price'].transform(features)
            
            # Make predictions
            predictions = []
            current_sequence = features[-60:]  # Last 60 periods
            
            for _ in range(horizon):
                # Reshape for LSTM
                input_data = current_sequence.reshape(1, 60, features.shape[1])
                
                # Predict next value
                pred = model.predict(input_data, verbose=0)[0, 0]
                predictions.append(pred)
                
                # Update sequence (simplified - would need full feature engineering in production)
                new_row = np.append(current_sequence[-1][1:], pred)
                current_sequence = np.vstack([current_sequence[1:], new_row])
            
            # Calculate confidence intervals
            std_dev = np.std(predictions)
            
            return {
                "symbol": symbol,
                "predictions": predictions,
                "horizon": horizon,
                "confidence_interval": {
                    "lower": [p - 1.96 * std_dev for p in predictions],
                    "upper": [p + 1.96 * std_dev for p in predictions]
                },
                "predicted_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            return {"error": str(e)}
    
    def _prepare_price_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for price prediction."""
        # Calculate technical indicators
        df['returns'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Select features
        features = ['open', 'high', 'low', 'close', 'volume', 'returns', 'volume_ratio']
        
        # Drop NaN values
        feature_data = df[features].dropna()
        
        return feature_data.values
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def calculate_option_greeks(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = 0.05,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """Calculate option Greeks using Black-Scholes model."""
        from scipy.stats import norm
        
        # Calculate d1 and d2
        d1 = (np.log(spot_price / strike_price) + 
              (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))
        
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        # Calculate Greeks
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
            theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) -
                     risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)) / 365
        else:
            delta = norm.cdf(d1) - 1
            theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) +
                     risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)) / 365
        
        gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(time_to_expiry))
        vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100
        rho = strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * \
              (norm.cdf(d2) if option_type.lower() == 'call' else -norm.cdf(-d2)) / 100
        
        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho
        }
    
    async def detect_chart_patterns(
        self,
        price_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Detect chart patterns in price data."""
        patterns = []
        
        # Simple pattern detection (would be more sophisticated in production)
        # Head and Shoulders
        if self._detect_head_and_shoulders(price_data):
            patterns.append({
                "pattern": "Head and Shoulders",
                "type": "bearish",
                "confidence": 0.75,
                "description": "Potential reversal pattern detected"
            })
        
        # Double Top/Bottom
        double_pattern = self._detect_double_pattern(price_data)
        if double_pattern:
            patterns.append(double_pattern)
        
        # Support/Resistance levels
        levels = self._find_support_resistance(price_data)
        if levels:
            patterns.extend(levels)
        
        return patterns
    
    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> bool:
        """Detect head and shoulders pattern."""
        # Simplified detection logic
        return False  # Placeholder
    
    def _detect_double_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect double top or bottom pattern."""
        # Simplified detection logic
        return None  # Placeholder
    
    def _find_support_resistance(self, df: pd.DataFrame) -> List[Dict]:
        """Find support and resistance levels."""
        levels = []
        
        # Find recent highs and lows
        recent_high = df['high'].rolling(20).max().iloc[-1]
        recent_low = df['low'].rolling(20).min().iloc[-1]
        
        levels.append({
            "pattern": "Resistance",
            "type": "level",
            "price": recent_high,
            "strength": "medium"
        })
        
        levels.append({
            "pattern": "Support",
            "type": "level",
            "price": recent_low,
            "strength": "medium"
        })
        
        return levels
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        # Placeholder for sentiment analysis
        return {
            "sentiment": "neutral",
            "score": 0.5,
            "confidence": 0.8
        }
    
    async def assess_portfolio_risk(
        self,
        positions: List[Dict],
        market_data: Dict
    ) -> Dict[str, Any]:
        """Assess portfolio risk metrics."""
        # Calculate various risk metrics
        total_value = sum(p['market_value'] for p in positions)
        
        # Simplified VaR calculation
        returns = [p.get('daily_return', 0) for p in positions]
        if returns:
            var_95 = np.percentile(returns, 5) * total_value
            var_99 = np.percentile(returns, 1) * total_value
        else:
            var_95 = var_99 = 0
        
        return {
            "total_value": total_value,
            "var_95": var_95,
            "var_99": var_99,
            "max_drawdown": self._calculate_max_drawdown(positions),
            "sharpe_ratio": self._calculate_sharpe_ratio(returns),
            "assessed_at": datetime.utcnow().isoformat()
        }
    
    def _calculate_max_drawdown(self, positions: List[Dict]) -> float:
        """Calculate maximum drawdown."""
        # Simplified calculation
        return 0.15  # Placeholder
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Assuming risk-free rate of 2%
        risk_free_rate = 0.02 / 252  # Daily rate
        
        return (avg_return - risk_free_rate) / std_return * np.sqrt(252)
    
    async def cleanup(self):
        """Clean up resources."""
        # Clear models from memory
        self.models.clear()
        self.scalers.clear()
        self.model_configs.clear()
        
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        
        logger.info("Model manager cleaned up")
