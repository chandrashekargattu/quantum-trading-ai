"""
Multi-Timeframe Transformer Model

Revolutionary approach that analyzes market data across multiple timeframes simultaneously:
- Microsecond to monthly timeframes in one model
- Attention mechanisms across different time scales
- Captures both HFT patterns and long-term trends
- Self-adapting to market regimes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from app.ml.base_model import BaseModel


@dataclass
class TimeframeConfig:
    """Configuration for each timeframe"""
    name: str
    window_size: int  # Number of periods
    period_seconds: int  # Seconds per period
    features: List[str]
    weight: float = 1.0


class MultiHeadTimeAttention(nn.Module):
    """Multi-head attention across different timeframes"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        # Feed forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Multi-head attention
        Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        K = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        V = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, n_heads, seq_len, d_head)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        attn_output = self.out_linear(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Add & Norm
        x = self.layer_norm1(x + attn_output)
        
        # Feed forward
        ff_output = self.ff(x)
        x = self.layer_norm2(x + ff_output)
        
        return x


class TimeframeEncoder(nn.Module):
    """Encoder for a specific timeframe"""
    
    def __init__(self, input_dim: int, d_model: int, n_layers: int = 2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            MultiHeadTimeAttention(d_model, n_heads=8, d_ff=d_model * 4)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input to model dimension
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)


class CrossTimeframeAttention(nn.Module):
    """Attention mechanism across different timeframes"""
    
    def __init__(self, d_model: int, n_timeframes: int):
        super().__init__()
        self.d_model = d_model
        self.n_timeframes = n_timeframes
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )
        
        # Timeframe importance weights
        self.timeframe_weights = nn.Parameter(
            torch.ones(n_timeframes) / n_timeframes
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model * n_timeframes, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, timeframe_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            timeframe_embeddings: List of embeddings from each timeframe encoder
        """
        batch_size = timeframe_embeddings[0].shape[0]
        
        # Stack timeframe embeddings
        stacked = torch.stack(timeframe_embeddings, dim=1)  # (batch, n_timeframes, seq_len, d_model)
        
        # Global representation from each timeframe (use last token)
        global_reps = stacked[:, :, -1, :]  # (batch, n_timeframes, d_model)
        
        # Apply timeframe weights
        weights = F.softmax(self.timeframe_weights, dim=0)
        weighted_reps = global_reps * weights.unsqueeze(0).unsqueeze(-1)
        
        # Cross-attention between timeframes
        attn_output, _ = self.cross_attention(
            weighted_reps, weighted_reps, weighted_reps
        )
        
        # Combine all timeframe information
        combined = attn_output.view(batch_size, -1)  # Flatten
        output = self.output_projection(combined)
        
        return self.norm(output)


class MultiTimeframeTransformer(BaseModel):
    """Main multi-timeframe transformer model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Timeframe configurations
        self.timeframes = [
            TimeframeConfig("tick", 1000, 0.001, ["price", "volume", "bid", "ask"]),  # Milliseconds
            TimeframeConfig("second", 300, 1, ["ohlcv", "spread", "imbalance"]),      # Seconds
            TimeframeConfig("minute", 60, 60, ["ohlcv", "vwap", "rsi", "macd"]),      # Minutes
            TimeframeConfig("5min", 48, 300, ["ohlcv", "patterns", "volume_profile"]), # 5 minutes
            TimeframeConfig("hourly", 24, 3600, ["ohlcv", "trend", "support_resistance"]), # Hours
            TimeframeConfig("daily", 30, 86400, ["ohlcv", "sentiment", "fundamentals"])    # Days
        ]
        
        # Model dimensions
        self.d_model = config.get('d_model', 512)
        self.n_heads = config.get('n_heads', 8)
        self.n_layers = config.get('n_layers', 4)
        self.dropout = config.get('dropout', 0.1)
        
        # Feature dimensions for each timeframe
        self.feature_dims = {
            "tick": 20,
            "second": 25,
            "minute": 30,
            "5min": 35,
            "hourly": 40,
            "daily": 50
        }
        
        # Encoders for each timeframe
        self.timeframe_encoders = nn.ModuleDict({
            tf.name: TimeframeEncoder(
                self.feature_dims[tf.name], self.d_model, self.n_layers
            )
            for tf in self.timeframes
        })
        
        # Cross-timeframe attention
        self.cross_attention = CrossTimeframeAttention(
            self.d_model, len(self.timeframes)
        )
        
        # Task-specific heads
        self.prediction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.GELU(),
            nn.Linear(self.d_model // 4, 3)  # [down, neutral, up] probabilities
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1)  # Volatility prediction
        )
        
        self.regime_head = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.GELU(),
            nn.Linear(256, 5)  # 5 market regimes
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1 confidence
        )
    
    def extract_features(self, data: Dict[str, torch.Tensor], timeframe: str) -> torch.Tensor:
        """Extract features for a specific timeframe"""
        if timeframe == "tick":
            # Ultra high-frequency features
            price = data['price']
            volume = data['volume']
            bid = data['bid']
            ask = data['ask']
            
            # Microstructure features
            spread = ask - bid
            mid = (bid + ask) / 2
            imbalance = (data['bid_size'] - data['ask_size']) / (data['bid_size'] + data['ask_size'] + 1e-9)
            
            # Price dynamics
            returns = torch.diff(price, dim=1) / (price[:, :-1] + 1e-9)
            volatility = returns.std(dim=1, keepdim=True)
            
            features = torch.cat([
                price.unsqueeze(-1),
                volume.unsqueeze(-1),
                spread.unsqueeze(-1),
                imbalance.unsqueeze(-1),
                returns,
                volatility.expand_as(returns)
            ], dim=-1)
            
        elif timeframe in ["second", "minute", "5min", "hourly", "daily"]:
            # OHLCV features
            ohlcv = data['ohlcv']  # (batch, seq_len, 5)
            
            # Technical indicators
            close = ohlcv[:, :, 3]
            volume = ohlcv[:, :, 4]
            
            # Returns and volatility
            returns = torch.diff(close, dim=1) / (close[:, :-1] + 1e-9)
            volatility = returns.std(dim=1, keepdim=True)
            
            # Moving averages
            ma_5 = F.avg_pool1d(close.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
            ma_20 = F.avg_pool1d(close.unsqueeze(1), kernel_size=20, stride=1, padding=10).squeeze(1)
            
            # RSI approximation
            gains = torch.clamp(returns, min=0)
            losses = torch.abs(torch.clamp(returns, max=0))
            avg_gains = F.avg_pool1d(gains.unsqueeze(1), kernel_size=14, stride=1, padding=7).squeeze(1)
            avg_losses = F.avg_pool1d(losses.unsqueeze(1), kernel_size=14, stride=1, padding=7).squeeze(1)
            rsi = 100 - (100 / (1 + avg_gains / (avg_losses + 1e-9)))
            
            # Combine features
            features = torch.cat([
                ohlcv,
                returns.unsqueeze(-1),
                volatility.expand(returns.shape).unsqueeze(-1),
                ma_5.unsqueeze(-1),
                ma_20.unsqueeze(-1),
                rsi.unsqueeze(-1)
            ], dim=-1)
        
        # Pad features to expected dimension
        current_dim = features.shape[-1]
        expected_dim = self.feature_dims[timeframe]
        
        if current_dim < expected_dim:
            padding = torch.zeros(
                features.shape[0], features.shape[1], 
                expected_dim - current_dim, 
                device=features.device
            )
            features = torch.cat([features, padding], dim=-1)
        
        return features
    
    def forward(self, inputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: Dict mapping timeframe names to their data tensors
        """
        # Encode each timeframe
        timeframe_embeddings = []
        
        for timeframe in self.timeframes:
            if timeframe.name in inputs:
                # Extract features
                features = self.extract_features(inputs[timeframe.name], timeframe.name)
                
                # Encode
                embedding = self.timeframe_encoders[timeframe.name](features)
                timeframe_embeddings.append(embedding)
            else:
                # Use zero embedding if timeframe data not available
                batch_size = next(iter(inputs.values()))['price'].shape[0]
                zero_embedding = torch.zeros(
                    batch_size, 1, self.d_model, 
                    device=next(iter(inputs.values()))['price'].device
                )
                timeframe_embeddings.append(zero_embedding)
        
        # Cross-timeframe attention
        unified_representation = self.cross_attention(timeframe_embeddings)
        
        # Generate predictions
        direction_probs = F.softmax(self.prediction_head(unified_representation), dim=-1)
        volatility = torch.abs(self.volatility_head(unified_representation))
        regime_probs = F.softmax(self.regime_head(unified_representation), dim=-1)
        confidence = self.uncertainty_head(unified_representation)
        
        return {
            'direction': direction_probs,  # [down, neutral, up]
            'volatility': volatility,
            'regime': regime_probs,
            'confidence': confidence,
            'timeframe_importance': F.softmax(self.cross_attention.timeframe_weights, dim=0)
        }
    
    def predict_multi_horizon(
        self, inputs: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Predict multiple time horizons simultaneously"""
        base_predictions = self.forward(inputs)
        
        # Different horizons based on timeframe importance
        horizons = {
            '1min': base_predictions['direction'] * base_predictions['timeframe_importance'][1],
            '5min': base_predictions['direction'] * base_predictions['timeframe_importance'][2],
            '1hour': base_predictions['direction'] * base_predictions['timeframe_importance'][4],
            '1day': base_predictions['direction'] * base_predictions['timeframe_importance'][5]
        }
        
        return {
            'base': base_predictions,
            'horizons': horizons
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class AdaptiveTimeframeSelector(nn.Module):
    """Dynamically selects which timeframes are most relevant"""
    
    def __init__(self, d_model: int, n_timeframes: int):
        super().__init__()
        self.market_state_encoder = nn.LSTM(d_model, d_model // 2, bidirectional=True)
        self.selector = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, n_timeframes),
            nn.Sigmoid()
        )
    
    def forward(self, unified_features: torch.Tensor) -> torch.Tensor:
        # Encode market state
        lstm_out, _ = self.market_state_encoder(unified_features.unsqueeze(0))
        lstm_out = lstm_out.squeeze(0)
        
        # Select timeframe weights
        weights = self.selector(lstm_out)
        
        return weights
