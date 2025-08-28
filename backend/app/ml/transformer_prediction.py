"""
Transformer-based Market Prediction Model

Implements state-of-the-art transformer architectures for financial market prediction,
including multi-modal data fusion and temporal attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import pandas as pd
from transformers import BertModel, BertTokenizer
import einops
from einops import rearrange, repeat

logger = logging.getLogger(__name__)


@dataclass
class MarketPredictionOutput:
    """Output from transformer market prediction model."""
    
    price_predictions: torch.Tensor  # Future price predictions
    volatility_predictions: torch.Tensor  # Volatility forecasts
    direction_probabilities: torch.Tensor  # Up/down/neutral probabilities
    attention_weights: torch.Tensor  # Attention weights for interpretability
    confidence_scores: torch.Tensor  # Model confidence in predictions
    feature_importance: Dict[str, float]  # Feature importance scores


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiModalEncoder(nn.Module):
    """
    Encoder for multiple data modalities including:
    - Price/volume data
    - Technical indicators
    - News/text data
    - Order book data
    - Alternative data sources
    """
    
    def __init__(
        self,
        price_dim: int,
        text_dim: int = 768,  # BERT embedding size
        order_book_dim: int = 50,
        d_model: int = 512
    ):
        super().__init__()
        
        # Price/volume encoder
        self.price_encoder = nn.Sequential(
            nn.Linear(price_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        
        # Text encoder (using pre-trained BERT)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_projection = nn.Linear(text_dim, d_model)
        
        # Order book encoder with CNN
        self.order_book_encoder = nn.Sequential(
            nn.Conv1d(order_book_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, d_model)
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1
        )
        
        # Modal fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model)
        )
    
    def forward(
        self,
        price_data: torch.Tensor,
        text_data: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        order_book_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Encode price data
        price_features = self.price_encoder(price_data)
        
        # Encode text data if available
        if text_data is not None:
            with torch.no_grad():
                bert_output = self.bert(
                    input_ids=text_data,
                    attention_mask=text_mask
                )
            text_features = self.text_projection(bert_output.pooler_output)
        else:
            text_features = torch.zeros_like(price_features)
        
        # Encode order book data if available
        if order_book_data is not None:
            order_book_features = self.order_book_encoder(
                order_book_data.transpose(1, 2)
            )
        else:
            order_book_features = torch.zeros_like(price_features)
        
        # Cross-modal attention
        attended_features, _ = self.cross_attention(
            price_features.unsqueeze(0),
            torch.stack([text_features, order_book_features]).transpose(0, 1),
            torch.stack([text_features, order_book_features]).transpose(0, 1)
        )
        attended_features = attended_features.squeeze(0)
        
        # Fuse all modalities
        fused_features = self.fusion(
            torch.cat([price_features, text_features, order_book_features], dim=-1)
        )
        
        return fused_features


class TemporalTransformer(nn.Module):
    """
    Temporal transformer for sequence modeling with:
    - Causal attention for autoregressive prediction
    - Multi-scale temporal attention
    - Adaptive computation time
    """
    
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
        dim_feedforward: int = 2048,
        max_seq_length: int = 1000
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Multi-scale temporal attention layers
        self.short_term_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout=0.1
            ),
            num_layers=num_layers // 3
        )
        
        self.medium_term_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout=0.1
            ),
            num_layers=num_layers // 3
        )
        
        self.long_term_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout=0.1
            ),
            num_layers=num_layers // 3
        )
        
        # Temporal fusion
        self.temporal_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model)
        )
        
        # Adaptive computation time
        self.halt_predictor = nn.Linear(d_model, 1)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Multi-scale processing
        # Short-term: last 10 time steps
        short_term = self.short_term_encoder(src[-10:], src_mask)
        
        # Medium-term: last 50 time steps
        medium_term = self.medium_term_encoder(src[-50:], src_mask)
        
        # Long-term: all time steps
        long_term = self.long_term_encoder(src, src_mask)
        
        # Align dimensions for fusion
        short_term_aligned = short_term[-1:].expand(src.size(0), -1, -1)
        medium_term_aligned = medium_term[-1:].expand(src.size(0), -1, -1)
        
        # Temporal fusion
        fused = self.temporal_fusion(
            torch.cat([short_term_aligned, medium_term_aligned, long_term], dim=-1)
        )
        
        # Adaptive computation (halting probabilities)
        halt_probs = torch.sigmoid(self.halt_predictor(fused))
        
        return fused, halt_probs


class MarketTransformer(nn.Module):
    """
    Complete market prediction transformer combining:
    - Multi-modal encoding
    - Temporal modeling
    - Multi-task prediction heads
    """
    
    def __init__(
        self,
        price_dim: int = 50,
        num_assets: int = 100,
        prediction_horizon: int = 5,
        d_model: int = 512,
        num_layers: int = 12
    ):
        super().__init__()
        
        self.num_assets = num_assets
        self.prediction_horizon = prediction_horizon
        
        # Multi-modal encoder
        self.modal_encoder = MultiModalEncoder(price_dim, d_model=d_model)
        
        # Temporal transformer
        self.temporal_transformer = TemporalTransformer(
            d_model=d_model,
            num_layers=num_layers
        )
        
        # Prediction heads
        self.price_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, prediction_horizon * num_assets)
        )
        
        self.volatility_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, prediction_horizon * num_assets)
        )
        
        self.direction_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 3 * num_assets)  # Up/Down/Neutral
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_assets)
        )
        
        # Feature importance through attention
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8
        )
    
    def forward(
        self,
        price_data: torch.Tensor,
        text_data: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        order_book_data: Optional[torch.Tensor] = None
    ) -> MarketPredictionOutput:
        batch_size = price_data.size(0)
        seq_len = price_data.size(1)
        
        # Encode each time step
        encoded_sequence = []
        for t in range(seq_len):
            encoded = self.modal_encoder(
                price_data[:, t],
                text_data[:, t] if text_data is not None else None,
                text_mask[:, t] if text_mask is not None else None,
                order_book_data[:, t] if order_book_data is not None else None
            )
            encoded_sequence.append(encoded)
        
        # Stack encoded sequence
        encoded_sequence = torch.stack(encoded_sequence)  # [seq_len, batch, d_model]
        
        # Temporal modeling
        temporal_features, halt_probs = self.temporal_transformer(encoded_sequence)
        
        # Take the last temporal feature for prediction
        last_features = temporal_features[-1]
        
        # Multi-task predictions
        price_predictions = self.price_predictor(last_features)
        price_predictions = rearrange(
            price_predictions,
            'b (h a) -> b h a',
            h=self.prediction_horizon,
            a=self.num_assets
        )
        
        volatility_predictions = self.volatility_predictor(last_features)
        volatility_predictions = rearrange(
            volatility_predictions,
            'b (h a) -> b h a',
            h=self.prediction_horizon,
            a=self.num_assets
        )
        volatility_predictions = F.softplus(volatility_predictions)  # Ensure positive
        
        direction_logits = self.direction_classifier(last_features)
        direction_logits = rearrange(
            direction_logits,
            'b (c a) -> b a c',
            c=3,
            a=self.num_assets
        )
        direction_probabilities = F.softmax(direction_logits, dim=-1)
        
        # Confidence scores
        confidence_scores = torch.sigmoid(self.confidence_estimator(last_features))
        
        # Feature importance through self-attention
        attended_features, attention_weights = self.feature_attention(
            last_features.unsqueeze(0),
            encoded_sequence,
            encoded_sequence
        )
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(
            attention_weights.squeeze(0)
        )
        
        return MarketPredictionOutput(
            price_predictions=price_predictions,
            volatility_predictions=volatility_predictions,
            direction_probabilities=direction_probabilities,
            attention_weights=attention_weights,
            confidence_scores=confidence_scores,
            feature_importance=feature_importance
        )
    
    def _calculate_feature_importance(
        self,
        attention_weights: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate feature importance from attention weights."""
        # Average attention across heads and batch
        avg_attention = attention_weights.mean(dim=(0, 1))
        
        # Map to feature names (simplified)
        feature_importance = {
            'price': float(avg_attention[:10].mean()),
            'volume': float(avg_attention[10:20].mean()),
            'technical_indicators': float(avg_attention[20:40].mean()),
            'news_sentiment': float(avg_attention[40:50].mean()),
            'order_book': float(avg_attention[50:].mean())
        }
        
        # Normalize
        total = sum(feature_importance.values())
        feature_importance = {
            k: v / total for k, v in feature_importance.items()
        }
        
        return feature_importance


class TransformerPredictionEngine:
    """
    Main engine for transformer-based market prediction.
    Handles data preprocessing, training, and inference.
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model_config = model_config
        
        # Initialize model
        self.model = MarketTransformer(**model_config).to(device)
        
        # Initialize tokenizer for text data
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Loss functions
        self.price_loss_fn = nn.HuberLoss()
        self.volatility_loss_fn = nn.MSELoss()
        self.direction_loss_fn = nn.CrossEntropyLoss()
    
    def preprocess_data(
        self,
        price_data: pd.DataFrame,
        news_data: Optional[pd.DataFrame] = None,
        order_book_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, torch.Tensor]:
        """Preprocess multi-modal market data."""
        # Normalize price data
        price_tensor = torch.tensor(
            price_data.values,
            dtype=torch.float32
        )
        price_tensor = (price_tensor - price_tensor.mean()) / price_tensor.std()
        
        # Process text data if available
        text_tensor = None
        text_mask = None
        if news_data is not None:
            encoded_news = []
            masks = []
            for text in news_data['text']:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                encoded_news.append(encoding['input_ids'])
                masks.append(encoding['attention_mask'])
            
            text_tensor = torch.cat(encoded_news)
            text_mask = torch.cat(masks)
        
        # Process order book data
        order_book_tensor = None
        if order_book_data is not None:
            order_book_tensor = torch.tensor(
                order_book_data.values,
                dtype=torch.float32
            )
            # Normalize
            order_book_tensor = (
                order_book_tensor - order_book_tensor.mean()
            ) / order_book_tensor.std()
        
        return {
            'price_data': price_tensor,
            'text_data': text_tensor,
            'text_mask': text_mask,
            'order_book_data': order_book_tensor
        }
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Forward pass
        predictions = self.model(
            price_data=batch['price_data'].to(self.device),
            text_data=batch.get('text_data', None),
            text_mask=batch.get('text_mask', None),
            order_book_data=batch.get('order_book_data', None)
        )
        
        # Calculate losses
        price_loss = self.price_loss_fn(
            predictions.price_predictions,
            targets['prices'].to(self.device)
        )
        
        volatility_loss = self.volatility_loss_fn(
            predictions.volatility_predictions,
            targets['volatilities'].to(self.device)
        )
        
        direction_loss = self.direction_loss_fn(
            predictions.direction_probabilities.reshape(-1, 3),
            targets['directions'].to(self.device).reshape(-1)
        )
        
        # Confidence-weighted loss
        confidence_weight = predictions.confidence_scores.mean()
        total_loss = (
            price_loss +
            0.5 * volatility_loss +
            0.3 * direction_loss
        ) / confidence_weight
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'total_loss': total_loss.item(),
            'price_loss': price_loss.item(),
            'volatility_loss': volatility_loss.item(),
            'direction_loss': direction_loss.item(),
            'confidence': confidence_weight.item()
        }
    
    def predict(
        self,
        market_data: Dict[str, Any],
        return_attention: bool = False
    ) -> Dict[str, np.ndarray]:
        """Make predictions on new market data."""
        self.model.eval()
        
        # Preprocess data
        processed_data = self.preprocess_data(
            market_data['prices'],
            market_data.get('news'),
            market_data.get('order_book')
        )
        
        with torch.no_grad():
            # Add batch dimension
            for key, value in processed_data.items():
                if value is not None:
                    processed_data[key] = value.unsqueeze(0).to(self.device)
            
            # Get predictions
            output = self.model(**processed_data)
        
        # Convert to numpy
        predictions = {
            'price_predictions': output.price_predictions.cpu().numpy(),
            'volatility_predictions': output.volatility_predictions.cpu().numpy(),
            'direction_probabilities': output.direction_probabilities.cpu().numpy(),
            'confidence_scores': output.confidence_scores.cpu().numpy(),
            'feature_importance': output.feature_importance
        }
        
        if return_attention:
            predictions['attention_weights'] = output.attention_weights.cpu().numpy()
        
        return predictions
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_config': self.model_config
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
