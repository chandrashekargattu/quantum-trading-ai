"""
Comprehensive tests for Transformer-based Market Prediction module.
Tests multi-modal encoding, temporal transformers, and predictions.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from app.ml.transformer_prediction import (
    MarketPredictionOutput,
    PositionalEncoding,
    MultiModalEncoder,
    TemporalTransformer,
    MarketTransformer,
    TransformerPredictionEngine
)


class TestPositionalEncoding:
    """Test positional encoding module."""
    
    def test_positional_encoding_initialization(self):
        """Test initialization with various parameters."""
        # Default initialization
        pe1 = PositionalEncoding(d_model=512)
        assert pe1.pe.shape == (5000, 1, 512)
        
        # Custom max length
        pe2 = PositionalEncoding(d_model=256, max_len=1000)
        assert pe2.pe.shape == (1000, 1, 256)
        
        # Small model dimension
        pe3 = PositionalEncoding(d_model=64)
        assert pe3.pe.shape == (5000, 1, 64)
    
    def test_positional_encoding_forward(self):
        """Test forward pass of positional encoding."""
        pe = PositionalEncoding(d_model=128)
        
        # Test different sequence lengths
        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(seq_len, 32, 128)  # seq_len, batch, d_model
            output = pe(x)
            
            assert output.shape == x.shape
            assert not torch.equal(output, x)  # Should be modified
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_positional_encoding_deterministic(self):
        """Test that positional encoding is deterministic."""
        pe = PositionalEncoding(d_model=256)
        x = torch.randn(100, 16, 256)
        
        output1 = pe(x)
        output2 = pe(x)
        
        assert torch.equal(output1, output2)


class TestMultiModalEncoder:
    """Test multi-modal data encoder."""
    
    @pytest.fixture
    def encoder(self):
        """Create multi-modal encoder instance."""
        with patch('app.ml.transformer_prediction.BertModel'):
            return MultiModalEncoder(
                price_dim=50,
                text_dim=768,
                order_book_dim=50,
                d_model=512
            )
    
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        with patch('app.ml.transformer_prediction.BertModel'):
            encoder = MultiModalEncoder(
                price_dim=100,
                text_dim=768,
                order_book_dim=40,
                d_model=256
            )
            
            # Check price encoder
            assert encoder.price_encoder[0].in_features == 100
            assert encoder.price_encoder[0].out_features == 256
            
            # Check text projection
            assert encoder.text_projection.in_features == 768
            assert encoder.text_projection.out_features == 256
            
            # Check order book encoder
            assert encoder.order_book_encoder[0].in_channels == 40
    
    def test_encoder_forward_all_modalities(self, encoder):
        """Test forward pass with all modalities."""
        batch_size = 16
        
        # Create inputs
        price_data = torch.randn(batch_size, 50)
        text_data = torch.randint(0, 1000, (batch_size, 128))
        text_mask = torch.ones(batch_size, 128)
        order_book_data = torch.randn(batch_size, 20, 50)
        
        # Mock BERT output
        encoder.bert.return_value = MagicMock(
            pooler_output=torch.randn(batch_size, 768)
        )
        
        # Forward pass
        output = encoder(price_data, text_data, text_mask, order_book_data)
        
        assert output.shape == (batch_size, 512)
        assert not torch.isnan(output).any()
    
    def test_encoder_forward_price_only(self, encoder):
        """Test forward pass with only price data."""
        batch_size = 8
        price_data = torch.randn(batch_size, 50)
        
        output = encoder(price_data)
        
        assert output.shape == (batch_size, 512)
        assert not torch.isnan(output).any()
    
    def test_encoder_missing_modalities(self, encoder):
        """Test encoder with missing modalities."""
        batch_size = 4
        price_data = torch.randn(batch_size, 50)
        
        # Only price data
        output1 = encoder(price_data, None, None, None)
        assert output1.shape == (batch_size, 512)
        
        # Price + order book
        order_book_data = torch.randn(batch_size, 10, 50)
        output2 = encoder(price_data, None, None, order_book_data)
        assert output2.shape == (batch_size, 512)
        
        # Should give different outputs
        assert not torch.equal(output1, output2)


class TestTemporalTransformer:
    """Test temporal transformer module."""
    
    @pytest.fixture
    def temporal_transformer(self):
        """Create temporal transformer instance."""
        return TemporalTransformer(
            d_model=512,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
            max_seq_length=1000
        )
    
    def test_temporal_transformer_initialization(self):
        """Test initialization with various parameters."""
        # Small model
        tt1 = TemporalTransformer(d_model=128, nhead=4, num_layers=3)
        assert tt1.d_model == 128
        
        # Large model
        tt2 = TemporalTransformer(
            d_model=1024,
            nhead=16,
            num_layers=12,
            dim_feedforward=4096
        )
        assert tt2.d_model == 1024
    
    def test_temporal_transformer_forward(self, temporal_transformer):
        """Test forward pass of temporal transformer."""
        seq_len = 100
        batch_size = 8
        d_model = 512
        
        src = torch.randn(seq_len, batch_size, d_model)
        output, halt_probs = temporal_transformer(src)
        
        assert output.shape == (seq_len, batch_size, d_model)
        assert halt_probs.shape == (seq_len, batch_size, 1)
        assert torch.all(halt_probs >= 0) and torch.all(halt_probs <= 1)
    
    def test_multi_scale_processing(self, temporal_transformer):
        """Test multi-scale temporal processing."""
        # Test with different sequence lengths
        for seq_len in [5, 15, 60, 200]:
            src = torch.randn(seq_len, 4, 512)
            output, _ = temporal_transformer(src)
            
            # Output should maintain shape
            assert output.shape == src.shape
            
            # Should process successfully regardless of length
            assert not torch.isnan(output).any()
    
    def test_temporal_transformer_with_mask(self, temporal_transformer):
        """Test temporal transformer with attention mask."""
        seq_len = 50
        batch_size = 4
        
        src = torch.randn(seq_len, batch_size, 512)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        
        output, halt_probs = temporal_transformer(src, mask)
        
        assert output.shape == src.shape
        assert not torch.isnan(output).any()


class TestMarketTransformer:
    """Test complete market prediction transformer."""
    
    @pytest.fixture
    def market_transformer(self):
        """Create market transformer instance."""
        with patch('app.ml.transformer_prediction.BertModel'):
            return MarketTransformer(
                price_dim=50,
                num_assets=10,
                prediction_horizon=5,
                d_model=256,
                num_layers=6
            )
    
    def test_market_transformer_initialization(self):
        """Test market transformer initialization."""
        with patch('app.ml.transformer_prediction.BertModel'):
            mt = MarketTransformer(
                price_dim=100,
                num_assets=20,
                prediction_horizon=10,
                d_model=512,
                num_layers=12
            )
            
            assert mt.num_assets == 20
            assert mt.prediction_horizon == 10
            
            # Check prediction heads
            assert mt.price_predictor[-1].out_features == 10 * 20  # horizon * assets
            assert mt.volatility_predictor[-1].out_features == 10 * 20
            assert mt.direction_classifier[-1].out_features == 3 * 20
    
    def test_market_transformer_forward(self, market_transformer):
        """Test complete forward pass."""
        batch_size = 4
        seq_len = 20
        
        # Create inputs
        price_data = torch.randn(batch_size, seq_len, 50)
        
        # Mock BERT for text encoding
        market_transformer.modal_encoder.bert.return_value = MagicMock(
            pooler_output=torch.randn(batch_size, 768)
        )
        
        # Forward pass
        output = market_transformer(price_data)
        
        assert isinstance(output, MarketPredictionOutput)
        assert output.price_predictions.shape == (batch_size, 5, 10)  # batch, horizon, assets
        assert output.volatility_predictions.shape == (batch_size, 5, 10)
        assert output.direction_probabilities.shape == (batch_size, 10, 3)  # batch, assets, 3
        assert output.confidence_scores.shape == (batch_size, 10)
        
        # Check volatility is positive
        assert torch.all(output.volatility_predictions > 0)
        
        # Check probabilities sum to 1
        prob_sums = output.direction_probabilities.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)
    
    def test_market_transformer_multi_modal(self, market_transformer):
        """Test with multiple modalities."""
        batch_size = 2
        seq_len = 10
        
        price_data = torch.randn(batch_size, seq_len, 50)
        text_data = torch.randint(0, 1000, (batch_size, seq_len, 128))
        text_mask = torch.ones(batch_size, seq_len, 128)
        order_book_data = torch.randn(batch_size, seq_len, 20, 50)
        
        market_transformer.modal_encoder.bert.return_value = MagicMock(
            pooler_output=torch.randn(batch_size, 768)
        )
        
        output = market_transformer(price_data, text_data, text_mask, order_book_data)
        
        assert isinstance(output, MarketPredictionOutput)
        assert 'price' in output.feature_importance
        assert 'news_sentiment' in output.feature_importance
        assert 'order_book' in output.feature_importance
    
    def test_feature_importance_calculation(self, market_transformer):
        """Test feature importance calculation."""
        # Create mock attention weights
        attention_weights = torch.randn(4, 8, 100, 100)  # batch, heads, seq, seq
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        feature_importance = market_transformer._calculate_feature_importance(
            attention_weights
        )
        
        assert isinstance(feature_importance, dict)
        assert sum(feature_importance.values()) == pytest.approx(1.0, rel=1e-5)
        assert all(0 <= v <= 1 for v in feature_importance.values())


class TestTransformerPredictionEngine:
    """Test main prediction engine."""
    
    @pytest.fixture
    def prediction_engine(self):
        """Create prediction engine instance."""
        with patch('app.ml.transformer_prediction.BertModel'):
            with patch('app.ml.transformer_prediction.BertTokenizer'):
                config = {
                    'price_dim': 50,
                    'num_assets': 5,
                    'prediction_horizon': 3,
                    'd_model': 256,
                    'num_layers': 4
                }
                return TransformerPredictionEngine(config)
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        return {
            'prices': pd.DataFrame(
                np.random.randn(100, 5).cumsum(axis=0) + 100,
                index=dates,
                columns=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
            ),
            'news': pd.DataFrame({
                'text': ['Positive news'] * 100,
                'timestamp': dates
            }),
            'order_book': pd.DataFrame(
                np.random.randn(100, 20),
                index=dates
            )
        }
    
    def test_engine_initialization(self, prediction_engine):
        """Test engine initialization."""
        assert prediction_engine.model is not None
        assert prediction_engine.optimizer is not None
        assert prediction_engine.scheduler is not None
        assert prediction_engine.device in ['cuda', 'cpu']
    
    def test_preprocess_data(self, prediction_engine, sample_market_data):
        """Test data preprocessing."""
        # Mock tokenizer
        prediction_engine.tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 102]]),
            'attention_mask': torch.tensor([[1, 1]])
        }
        
        processed = prediction_engine.preprocess_data(
            sample_market_data['prices'],
            sample_market_data['news'],
            sample_market_data['order_book']
        )
        
        assert 'price_data' in processed
        assert 'text_data' in processed
        assert 'text_mask' in processed
        assert 'order_book_data' in processed
        
        # Check normalization
        assert processed['price_data'].mean().abs() < 1.0
        assert processed['price_data'].std() > 0.5
    
    def test_train_step(self, prediction_engine):
        """Test single training step."""
        batch_size = 4
        seq_len = 10
        
        # Create batch
        batch = {
            'price_data': torch.randn(batch_size, seq_len, 50)
        }
        
        # Create targets
        targets = {
            'prices': torch.randn(batch_size, 3, 5),  # horizon, assets
            'volatilities': torch.abs(torch.randn(batch_size, 3, 5)),
            'directions': torch.randint(0, 3, (batch_size, 5))
        }
        
        # Mock model output
        mock_output = MarketPredictionOutput(
            price_predictions=torch.randn(batch_size, 3, 5),
            volatility_predictions=torch.abs(torch.randn(batch_size, 3, 5)),
            direction_probabilities=torch.softmax(torch.randn(batch_size, 5, 3), dim=-1),
            attention_weights=torch.randn(1, 1, 10, 10),
            confidence_scores=torch.sigmoid(torch.randn(batch_size, 5)),
            feature_importance={'price': 1.0}
        )
        
        with patch.object(prediction_engine.model, 'forward', return_value=mock_output):
            losses = prediction_engine.train_step(batch, targets)
        
        assert 'total_loss' in losses
        assert 'price_loss' in losses
        assert 'volatility_loss' in losses
        assert 'direction_loss' in losses
        assert 'confidence' in losses
        
        # All losses should be positive
        assert all(v > 0 for v in losses.values())
    
    def test_predict(self, prediction_engine, sample_market_data):
        """Test prediction functionality."""
        # Mock tokenizer
        prediction_engine.tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 102]]),
            'attention_mask': torch.tensor([[1, 1]])
        }
        
        # Mock model output
        mock_output = MarketPredictionOutput(
            price_predictions=torch.randn(1, 3, 5),
            volatility_predictions=torch.abs(torch.randn(1, 3, 5)),
            direction_probabilities=torch.softmax(torch.randn(1, 5, 3), dim=-1),
            attention_weights=torch.randn(1, 1, 10, 10),
            confidence_scores=torch.sigmoid(torch.randn(1, 5)),
            feature_importance={'price': 0.6, 'news': 0.3, 'order_book': 0.1}
        )
        
        with patch.object(prediction_engine.model, 'forward', return_value=mock_output):
            predictions = prediction_engine.predict(sample_market_data)
        
        assert 'price_predictions' in predictions
        assert 'volatility_predictions' in predictions
        assert 'direction_probabilities' in predictions
        assert 'confidence_scores' in predictions
        assert 'feature_importance' in predictions
        
        # Check shapes
        assert predictions['price_predictions'].shape == (1, 3, 5)
        assert predictions['volatility_predictions'].shape == (1, 3, 5)
        assert predictions['direction_probabilities'].shape == (1, 5, 3)
        
        # Test with attention weights
        predictions_with_attention = prediction_engine.predict(
            sample_market_data,
            return_attention=True
        )
        assert 'attention_weights' in predictions_with_attention
    
    def test_save_load_model(self, prediction_engine, tmp_path):
        """Test model saving and loading."""
        # Save model
        save_path = tmp_path / "transformer_model.pth"
        prediction_engine.save_model(str(save_path))
        
        assert save_path.exists()
        
        # Create new engine and load
        with patch('app.ml.transformer_prediction.BertModel'):
            with patch('app.ml.transformer_prediction.BertTokenizer'):
                new_engine = TransformerPredictionEngine(prediction_engine.model_config)
                new_engine.load_model(str(save_path))
        
        # Check model loaded correctly
        # Compare some parameters
        for p1, p2 in zip(
            prediction_engine.model.parameters(),
            new_engine.model.parameters()
        ):
            assert torch.equal(p1, p2)
    
    @pytest.mark.parametrize("price_dim,num_assets,horizon", [
        (30, 3, 1),
        (50, 5, 5),
        (100, 10, 10),
    ])
    def test_different_configurations(self, price_dim, num_assets, horizon):
        """Test engine with different configurations."""
        with patch('app.ml.transformer_prediction.BertModel'):
            with patch('app.ml.transformer_prediction.BertTokenizer'):
                config = {
                    'price_dim': price_dim,
                    'num_assets': num_assets,
                    'prediction_horizon': horizon,
                    'd_model': 128,
                    'num_layers': 2
                }
                engine = TransformerPredictionEngine(config)
                
                # Test prediction shapes
                mock_output = MarketPredictionOutput(
                    price_predictions=torch.randn(1, horizon, num_assets),
                    volatility_predictions=torch.abs(torch.randn(1, horizon, num_assets)),
                    direction_probabilities=torch.softmax(torch.randn(1, num_assets, 3), dim=-1),
                    attention_weights=torch.randn(1, 1, 10, 10),
                    confidence_scores=torch.sigmoid(torch.randn(1, num_assets)),
                    feature_importance={'price': 1.0}
                )
                
                assert engine.model.prediction_horizon == horizon
                assert engine.model.num_assets == num_assets
