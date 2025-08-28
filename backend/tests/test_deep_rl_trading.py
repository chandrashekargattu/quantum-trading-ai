"""
Comprehensive tests for Deep Reinforcement Learning Trading module.
Tests PPO agent, trading environment, and neural networks.
"""

import pytest
import numpy as np
import torch
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from collections import deque
import gym

from app.ml.deep_rl_trading import (
    TradingState,
    AttentionNetwork,
    DeepTradingNetwork,
    PPOAgent,
    TradingEnvironment,
    DeepRLTradingEngine
)


class TestTradingState:
    """Test trading state representation."""
    
    def test_trading_state_initialization(self):
        """Test TradingState creation and tensor conversion."""
        # Create sample state
        state = TradingState(
            prices=np.array([100, 101, 102, 103, 104]),
            volumes=np.array([1000, 1100, 1200, 1300, 1400]),
            technical_indicators=np.random.randn(20),
            order_book=np.random.randn(10),
            portfolio_state=np.array([10000, 50, 10500]),
            market_microstructure=np.random.randn(15),
            sentiment_scores=np.random.randn(5),
            volatility_surface=np.random.randn(10, 10),
            correlation_matrix=np.eye(5)
        )
        
        # Test tensor conversion
        tensor = state.to_tensor()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert len(tensor.shape) == 1  # Should be flattened
        
    def test_trading_state_edge_cases(self):
        """Test TradingState with edge cases."""
        # Empty arrays
        state = TradingState(
            prices=np.array([]),
            volumes=np.array([]),
            technical_indicators=np.array([]),
            order_book=np.array([]),
            portfolio_state=np.array([]),
            market_microstructure=np.array([]),
            sentiment_scores=np.array([]),
            volatility_surface=np.array([[]]),
            correlation_matrix=np.array([[]])
        )
        
        tensor = state.to_tensor()
        assert tensor.numel() == 0  # Empty tensor
        
        # Single values
        state_single = TradingState(
            prices=np.array([100]),
            volumes=np.array([1000]),
            technical_indicators=np.array([0.5]),
            order_book=np.array([1.0]),
            portfolio_state=np.array([10000]),
            market_microstructure=np.array([0.1]),
            sentiment_scores=np.array([0.7]),
            volatility_surface=np.array([[0.2]]),
            correlation_matrix=np.array([[1.0]])
        )
        
        tensor_single = state_single.to_tensor()
        assert tensor_single.numel() == 8  # Total elements


class TestAttentionNetwork:
    """Test multi-head attention network."""
    
    @pytest.fixture
    def attention_net(self):
        """Create attention network instance."""
        return AttentionNetwork(input_dim=128, hidden_dim=64, num_heads=8)
    
    def test_attention_initialization(self):
        """Test attention network initialization."""
        net = AttentionNetwork(input_dim=256, hidden_dim=128, num_heads=4)
        
        # Check layer dimensions
        assert net.query.in_features == 256
        assert net.query.out_features == 128 * 4
        assert net.num_heads == 4
        assert net.hidden_dim == 128
        
    def test_attention_forward(self, attention_net):
        """Test attention forward pass."""
        batch_size = 32
        seq_len = 10
        input_dim = 128
        
        # Create input
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output = attention_net(x)
        
        # Check output shape
        assert output.shape == (batch_size, 64)  # hidden_dim
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
    def test_attention_gradient_flow(self, attention_net):
        """Test gradient flow through attention."""
        x = torch.randn(16, 5, 128, requires_grad=True)
        output = attention_net(x)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check parameter gradients
        for param in attention_net.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestDeepTradingNetwork:
    """Test deep trading neural network."""
    
    @pytest.fixture
    def trading_net(self):
        """Create trading network instance."""
        return DeepTradingNetwork(state_dim=100, action_dim=10, hidden_dim=256)
    
    def test_network_initialization(self):
        """Test network initialization with various parameters."""
        # Default initialization
        net1 = DeepTradingNetwork(state_dim=50, action_dim=5)
        assert net1.lstm.input_size == 50
        assert net1.action_head.out_features == 5
        assert net1.value_head.out_features == 1
        
        # Custom parameters
        net2 = DeepTradingNetwork(
            state_dim=200,
            action_dim=20,
            hidden_dim=512,
            num_layers=8
        )
        assert net2.lstm.hidden_size == 512
        
    def test_network_forward(self, trading_net):
        """Test network forward pass."""
        batch_size = 16
        state = torch.randn(batch_size, 100)
        
        # Without hidden state
        action_logits, value, hidden = trading_net(state)
        
        assert action_logits.shape == (batch_size, 10)
        assert value.shape == (batch_size, 1)
        assert isinstance(hidden, tuple)
        assert len(hidden) == 2  # (h, c) for LSTM
        
        # With hidden state
        action_logits2, value2, hidden2 = trading_net(state, hidden)
        assert not torch.equal(hidden[0], hidden2[0])  # Hidden state updated
        
    def test_network_edge_cases(self, trading_net):
        """Test network with edge cases."""
        # Single sample
        state_single = torch.randn(1, 100)
        action_logits, value, hidden = trading_net(state_single)
        assert action_logits.shape == (1, 10)
        
        # Large batch
        state_large = torch.randn(1000, 100)
        action_logits, value, hidden = trading_net(state_large)
        assert action_logits.shape == (1000, 10)
        
        # Check numerical stability
        assert not torch.isnan(action_logits).any()
        assert not torch.isinf(action_logits).any()


class TestPPOAgent:
    """Test Proximal Policy Optimization agent."""
    
    @pytest.fixture
    def ppo_agent(self):
        """Create PPO agent instance."""
        return PPOAgent(state_dim=100, action_dim=10, lr=3e-4)
    
    @pytest.fixture
    def sample_state(self):
        """Create sample trading state."""
        return TradingState(
            prices=np.random.randn(50),
            volumes=np.random.randn(50),
            technical_indicators=np.random.randn(20),
            order_book=np.random.randn(10),
            portfolio_state=np.array([10000, 50, 10500]),
            market_microstructure=np.random.randn(15),
            sentiment_scores=np.random.randn(5),
            volatility_surface=np.random.randn(10, 10),
            correlation_matrix=np.eye(5)
        )
    
    def test_ppo_initialization(self):
        """Test PPO agent initialization."""
        agent = PPOAgent(
            state_dim=200,
            action_dim=20,
            lr=1e-3,
            gamma=0.95,
            eps_clip=0.1,
            k_epochs=10
        )
        
        assert agent.gamma == 0.95
        assert agent.eps_clip == 0.1
        assert agent.k_epochs == 10
        
        # Check both policies initialized
        assert agent.policy is not None
        assert agent.policy_old is not None
        
        # Check they start with same weights
        for p1, p2 in zip(agent.policy.parameters(), agent.policy_old.parameters()):
            assert torch.equal(p1, p2)
    
    def test_select_action(self, ppo_agent, sample_state):
        """Test action selection."""
        # Select action
        action, hidden = ppo_agent.select_action(sample_state)
        
        assert isinstance(action, int)
        assert 0 <= action < 10  # Within action space
        assert hidden is not None
        
        # Multiple selections should give different actions (stochastic)
        actions = [ppo_agent.select_action(sample_state)[0] for _ in range(100)]
        assert len(set(actions)) > 1  # Should have variety
    
    def test_ppo_update(self, ppo_agent, sample_state):
        """Test PPO update mechanism."""
        # Collect some experiences
        for _ in range(150):  # More than minimum required
            state_tensor = sample_state.to_tensor()
            action = np.random.randint(0, 10)
            reward = np.random.randn()
            log_prob = torch.tensor([0.0])
            
            ppo_agent.memory.append((state_tensor, action, reward, log_prob))
        
        # Get initial policy parameters
        initial_params = [p.clone() for p in ppo_agent.policy.parameters()]
        
        # Update
        ppo_agent.update()
        
        # Check parameters changed
        for p1, p2 in zip(initial_params, ppo_agent.policy.parameters()):
            assert not torch.equal(p1, p2)
        
        # Check memory cleared
        assert len(ppo_agent.memory) == 0
        
        # Check old policy updated
        for p1, p2 in zip(ppo_agent.policy.parameters(), ppo_agent.policy_old.parameters()):
            assert torch.equal(p1, p2)
    
    def test_ppo_update_insufficient_memory(self, ppo_agent):
        """Test PPO update with insufficient memory."""
        # Add only a few experiences
        for _ in range(50):  # Less than required 100
            ppo_agent.memory.append((torch.randn(100), 0, 0.0, torch.tensor([0.0])))
        
        # Update should not happen
        initial_params = [p.clone() for p in ppo_agent.policy.parameters()]
        ppo_agent.update()
        
        # Parameters should remain unchanged
        for p1, p2 in zip(initial_params, ppo_agent.policy.parameters()):
            assert torch.equal(p1, p2)


class TestTradingEnvironment:
    """Test trading environment simulation."""
    
    @pytest.fixture
    def market_data(self):
        """Generate sample market data."""
        dates = pd.date_range('2023-01-01', periods=1000, freq='1min')
        return {
            'AAPL': np.random.randn(1000).cumsum() + 150,
            'GOOGL': np.random.randn(1000).cumsum() + 2800,
            'MSFT': np.random.randn(1000).cumsum() + 300
        }
    
    @pytest.fixture
    def trading_env(self, market_data):
        """Create trading environment."""
        return TradingEnvironment(
            market_data=market_data,
            initial_capital=100000,
            transaction_cost=0.001,
            max_position_size=0.1,
            leverage=2.0
        )
    
    def test_environment_initialization(self, trading_env):
        """Test environment initialization."""
        assert trading_env.initial_capital == 100000
        assert trading_env.transaction_cost == 0.001
        assert trading_env.max_position_size == 0.1
        assert trading_env.leverage == 2.0
        
        # Check action/observation spaces
        assert isinstance(trading_env.action_space, gym.spaces.MultiDiscrete)
        assert isinstance(trading_env.observation_space, gym.spaces.Box)
    
    def test_environment_reset(self, trading_env):
        """Test environment reset."""
        initial_state = trading_env.reset()
        
        assert isinstance(initial_state, TradingState)
        assert trading_env.current_step == 0
        assert trading_env.capital == trading_env.initial_capital
        assert all(pos == 0 for pos in trading_env.positions.values())
        assert len(trading_env.trades) == 0
    
    def test_environment_step(self, trading_env):
        """Test environment step function."""
        trading_env.reset()
        
        # Take a buy action
        action = np.array([0, 0, 0, 5, 5, 5])  # Buy actions with 50% size
        state, reward, done, info = trading_env.step(action)
        
        assert isinstance(state, TradingState)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check info contains expected keys
        assert 'portfolio_value' in info
        assert 'positions' in info
        assert 'sharpe_ratio' in info
        assert 'max_drawdown' in info
    
    def test_environment_buy_sell_mechanics(self, trading_env, market_data):
        """Test buy and sell order execution."""
        trading_env.reset()
        
        # Buy order
        initial_capital = trading_env.capital
        trading_env._execute_buy('AAPL', 0.5)
        
        assert trading_env.positions['AAPL'] > 0
        assert trading_env.capital < initial_capital
        assert len(trading_env.trades) == 1
        
        # Sell order
        initial_position = trading_env.positions['AAPL']
        trading_env._execute_sell('AAPL', 0.5)
        
        assert trading_env.positions['AAPL'] < initial_position
        assert len(trading_env.trades) == 2
    
    def test_environment_market_impact(self, trading_env):
        """Test market impact modeling."""
        trading_env.reset()
        
        # Large order should have higher impact
        small_order_capital = trading_env.capital
        trading_env._execute_buy('AAPL', 0.1)  # Small order
        small_order_cost = small_order_capital - trading_env.capital
        
        trading_env.reset()
        large_order_capital = trading_env.capital
        trading_env._execute_buy('AAPL', 1.0)  # Large order
        large_order_cost = large_order_capital - trading_env.capital
        
        # Large order should cost more per share due to impact
        assert large_order_cost > small_order_cost * 10
    
    def test_environment_position_limits(self, trading_env):
        """Test position limits enforcement."""
        trading_env.reset()
        
        # Try to exceed position limit
        max_position_value = trading_env.capital * trading_env.max_position_size
        
        # Execute maximum allowed position
        trading_env._execute_buy('AAPL', 1.0)
        position_value = trading_env.positions['AAPL'] * trading_env.market_data['AAPL'][0]
        
        assert position_value <= max_position_value * trading_env.leverage
    
    def test_environment_done_conditions(self, trading_env):
        """Test episode termination conditions."""
        trading_env.reset()
        
        # Test time limit
        trading_env.current_step = len(trading_env.market_data['AAPL']) - 1
        _, _, done, _ = trading_env.step(np.zeros(6))
        assert done
        
        # Test drawdown stop
        trading_env.reset()
        trading_env.portfolio_values = [100000, 40000]  # 60% drawdown
        _, _, done, _ = trading_env.step(np.zeros(6))
        assert done
    
    def test_reward_calculation(self, trading_env):
        """Test reward function calculation."""
        trading_env.reset()
        
        # Positive return should give positive reward
        trading_env.portfolio_values = [100000, 101000]
        reward = trading_env._calculate_reward()
        assert reward > 0
        
        # Negative return with high drawdown should give negative reward
        trading_env.portfolio_values = [100000, 95000, 90000]
        reward = trading_env._calculate_reward()
        assert reward < 0


class TestDeepRLTradingEngine:
    """Test main RL trading engine."""
    
    @pytest.fixture
    def trading_engine(self, market_data):
        """Create trading engine."""
        return DeepRLTradingEngine(
            market_data=market_data,
            agent_type='PPO',
            hyperparameters={'lr': 3e-4}
        )
    
    def test_engine_initialization(self, trading_engine):
        """Test engine initialization."""
        assert trading_engine.agent_type == 'PPO'
        assert trading_engine.agent is not None
        assert trading_engine.env is not None
        assert isinstance(trading_engine.training_history, dict)
    
    def test_engine_unsupported_agent(self, market_data):
        """Test engine with unsupported agent type."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            DeepRLTradingEngine(
                market_data=market_data,
                agent_type='INVALID'
            )
    
    @patch('app.ml.deep_rl_trading.PPOAgent.select_action')
    @patch('app.ml.deep_rl_trading.PPOAgent.update')
    def test_engine_train(self, mock_update, mock_select_action, trading_engine):
        """Test training loop."""
        # Mock action selection
        mock_select_action.return_value = (0, None)
        
        # Train for few episodes
        trading_engine.train(n_episodes=2, save_interval=10)
        
        # Check training history updated
        assert len(trading_engine.training_history['rewards']) == 2
        assert len(trading_engine.training_history['portfolio_values']) == 2
        assert len(trading_engine.training_history['sharpe_ratios']) == 2
        assert len(trading_engine.training_history['max_drawdowns']) == 2
        
        # Check agent update called
        assert mock_update.called
    
    def test_engine_backtest(self, trading_engine, market_data):
        """Test backtesting functionality."""
        # Create test data
        test_data = {
            'AAPL': market_data['AAPL'][:100],
            'GOOGL': market_data['GOOGL'][:100],
            'MSFT': market_data['MSFT'][:100]
        }
        
        # Run backtest
        results = trading_engine.backtest(test_data)
        
        # Check results structure
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'win_rate' in results
        assert 'portfolio_values' in results
        assert 'trades' in results
        assert 'final_positions' in results
        
        # Validate metrics
        assert isinstance(results['total_return'], float)
        assert isinstance(results['sharpe_ratio'], float)
        assert 0 <= results['win_rate'] <= 1
        assert results['max_drawdown'] >= 0
    
    def test_model_save_load(self, trading_engine, tmp_path):
        """Test model saving and loading."""
        # Train briefly
        trading_engine.train(n_episodes=1)
        
        # Save model
        save_path = tmp_path / "test_model.pth"
        trading_engine.save_model(str(save_path))
        
        assert save_path.exists()
        
        # Load model
        new_engine = DeepRLTradingEngine(
            market_data=trading_engine.market_data,
            agent_type='PPO'
        )
        new_engine.load_model(str(save_path))
        
        # Check state loaded
        assert len(new_engine.training_history['rewards']) > 0
    
    def test_calculate_max_drawdown(self, trading_engine):
        """Test maximum drawdown calculation."""
        # Test with various portfolio value sequences
        test_cases = [
            ([100, 110, 105, 120, 100], 16.67),  # ~16.67% drawdown
            ([100, 90, 80, 90, 100], 20.0),  # 20% drawdown
            ([100, 110, 120, 130], 0.0),  # No drawdown
            ([100], 0.0),  # Single value
        ]
        
        for values, expected_dd in test_cases:
            dd = trading_engine._calculate_max_drawdown(values)
            assert abs(dd - expected_dd) < 0.1  # Allow small numerical error
    
    @pytest.mark.parametrize("n_episodes,save_interval", [
        (10, 5),
        (20, 10),
        (5, 2),
    ])
    def test_training_parameters(self, trading_engine, n_episodes, save_interval):
        """Test training with different parameters."""
        with patch.object(trading_engine, 'save_model'):
            trading_engine.train(n_episodes=n_episodes, save_interval=save_interval)
            
            # Check correct number of saves
            expected_saves = n_episodes // save_interval
            assert trading_engine.save_model.call_count == expected_saves
