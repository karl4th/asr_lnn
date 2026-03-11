"""
Unit tests for Coordinated DREAMStack.

Tests for:
- CoordinatedDREAMStack
- UncoordinatedDREAMStack
- Top-down modulation
- Inter-layer prediction
- Hierarchical sleep

Run:
    uv run pytest tests/test_coordination.py -v
"""

import pytest
import torch
from dream import (
    CoordinatedDREAMStack,
    UncoordinatedDREAMStack,
    DREAMCell,
    DREAMConfig
)


class TestCoordinatedDREAMStack:
    """Tests for CoordinatedDREAMStack."""

    @pytest.fixture
    def coordinated_stack(self):
        """Create coordinated stack."""
        return CoordinatedDREAMStack(
            input_dim=80,
            hidden_dims=[128, 128, 128],
            rank=16,
            dropout=0.1
        )

    @pytest.fixture
    def uncoordinated_stack(self):
        """Create uncoordinated stack."""
        return UncoordinatedDREAMStack(
            input_dim=80,
            hidden_dims=[128, 128, 128],
            rank=16,
            dropout=0.1
        )

    def test_init(self, coordinated_stack):
        """Test initialization."""
        assert coordinated_stack.num_layers == 3
        assert len(coordinated_stack.layers) == 3
        assert coordinated_stack.input_dim == 80
        assert coordinated_stack.hidden_dims == [128, 128, 128]

    def test_forward_pass(self, coordinated_stack):
        """Test forward pass."""
        batch_size = 2
        time_steps = 50
        x = torch.randn(batch_size, time_steps, 80)

        output, states, coord_info = coordinated_stack(x)

        assert output.shape == (batch_size, 128)
        assert len(states) == 3
        assert isinstance(coord_info, dict)

    def test_forward_with_return_all(self, coordinated_stack):
        """Test forward with return_all=True."""
        batch_size = 2
        time_steps = 50
        x = torch.randn(batch_size, time_steps, 80)

        output, states, coord_info = coordinated_stack(x, return_all=True)

        assert output.shape == (batch_size, time_steps, 128)

    def test_coordination_info(self, coordinated_stack):
        """Test coordination info is returned."""
        batch_size = 2
        time_steps = 50
        x = torch.randn(batch_size, time_steps, 80)

        output, states, coord_info = coordinated_stack(x, return_all=True)

        assert 'modulations' in coord_info
        assert 'inter_layer_errors' in coord_info

    def test_parameter_count(self, coordinated_stack, uncoordinated_stack):
        """Test both stacks have parameters."""
        coord_params = coordinated_stack.count_parameters()
        uncoord_params = uncoordinated_stack.count_parameters()

        # Both should have reasonable parameter counts
        assert coord_params > 0
        assert uncoord_params > 0
        print(f"Coordinated: {coord_params:,}, Uncoordinated: {uncoord_params:,}")

    def test_init_states(self, coordinated_stack):
        """Test state initialization."""
        batch_size = 4
        states = coordinated_stack.init_states(batch_size)

        assert len(states) == 3
        assert all(s.h.shape == (batch_size, 128) for s in states)


class TestUncoordinatedDREAMStack:
    """Tests for UncoordinatedDREAMStack."""

    @pytest.fixture
    def stack(self):
        """Create uncoordinated stack."""
        return UncoordinatedDREAMStack(
            input_dim=80,
            hidden_dims=[64, 64],
            rank=8,
            dropout=0.0
        )

    def test_forward_pass(self, stack):
        """Test forward pass."""
        batch_size = 2
        time_steps = 50
        x = torch.randn(batch_size, time_steps, 80)

        output, states = stack(x)

        assert output.shape == (batch_size, 64)
        assert len(states) == 2

    def test_forward_with_return_all(self, stack):
        """Test forward with return_all=True."""
        batch_size = 2
        time_steps = 50
        x = torch.randn(batch_size, time_steps, 80)

        output, states = stack(x, return_all=True)

        assert output.shape == (batch_size, time_steps, 64)

    def test_no_coordination_in_cells(self, stack):
        """Test that cells don't have coordination."""
        for layer in stack.layers:
            assert not layer.use_coordination
            assert not hasattr(layer, 'W_mod') or layer.W_pred.numel() == 1


class TestTopDownModulation:
    """Tests for top-down modulation."""

    @pytest.fixture
    def cell_with_coordination(self):
        """Create cell with coordination."""
        config = DREAMConfig(
            input_dim=80,
            hidden_dim=128,
            rank=16,
            use_coordination=True
        )
        return DREAMCell(config)

    @pytest.fixture
    def cell_without_coordination(self):
        """Create cell without coordination."""
        config = DREAMConfig(
            input_dim=80,
            hidden_dim=128,
            rank=16,
            use_coordination=False
        )
        return DREAMCell(config)

    def test_generate_modulation(self, cell_with_coordination):
        """Test modulation generation."""
        batch_size = 2
        h = torch.randn(batch_size, 128)

        modulation = cell_with_coordination.generate_modulation(h)

        assert modulation.shape == (batch_size, 128)
        assert modulation.min() >= 0
        assert modulation.max() <= 1

    def test_generate_modulation_disabled(self, cell_without_coordination):
        """Test modulation returns ones when disabled."""
        batch_size = 2
        h = torch.randn(batch_size, 128)

        modulation = cell_without_coordination.generate_modulation(h)

        assert modulation.shape == (batch_size, 128)
        assert torch.allclose(modulation, torch.ones_like(modulation))

    def test_predict_lower_activity(self, cell_with_coordination):
        """Test inter-layer prediction."""
        batch_size = 2
        h = torch.randn(batch_size, 128)

        prediction = cell_with_coordination.predict_lower_activity(h)

        assert prediction.shape == (batch_size, 128)

    def test_compute_inter_layer_error(self, cell_with_coordination):
        """Test inter-layer error computation."""
        batch_size = 2
        prediction = torch.randn(batch_size, 128)
        actual = torch.randn(batch_size, 128)

        error = cell_with_coordination.compute_inter_layer_error(prediction, actual)

        assert error.shape == (batch_size, 128)
        assert torch.allclose(error, actual - prediction)

    def test_surprise_with_modulation(self, cell_with_coordination):
        """Test surprise computation with modulation."""
        from dream import DREAMState

        batch_size = 2
        config = cell_with_coordination.config

        state = DREAMState.init_from_config(config, batch_size)
        error = torch.randn(batch_size, 80)

        # Without modulation
        surprise_no_mod, _ = cell_with_coordination.compute_surprise(error, state)

        # With modulation (should lower threshold, increase surprise)
        modulation = torch.ones(batch_size, 128) * 0.9  # High modulation
        surprise_with_mod, _ = cell_with_coordination.compute_surprise(
            error, state, modulation
        )

        # With high modulation, threshold is lower, so surprise should be higher
        assert surprise_with_mod.mean() >= surprise_no_mod.mean()


class TestHierarchicalSleep:
    """Tests for hierarchical sleep consolidation."""

    @pytest.fixture
    def stack(self):
        """Create coordinated stack."""
        return CoordinatedDREAMStack(
            input_dim=80,
            hidden_dims=[64, 64],
            rank=8,
            dropout=0.0
        )

    def test_forward_with_global_sleep(self, stack):
        """Test forward with global sleep."""
        batch_size = 2
        time_steps = 50
        x = torch.randn(batch_size, time_steps, 80)

        output, states, global_surprise = stack.forward_with_global_sleep(x)

        assert output.shape == (batch_size, 64)
        assert isinstance(global_surprise, float)
        assert global_surprise >= 0

    def test_global_surprise_tracking(self, stack):
        """Test that global surprise is tracked."""
        batch_size = 2
        time_steps = 100
        x = torch.randn(batch_size, time_steps, 80)

        output, states, global_surprise = stack.forward_with_global_sleep(x)

        # Global surprise should be average across layers and time
        assert global_surprise >= 0
        assert global_surprise <= 1  # Should be normalized


class TestDeepStack:
    """Tests for deep stacks (4+ layers)."""

    @pytest.fixture
    def deep_stack(self):
        """Create deep coordinated stack."""
        return CoordinatedDREAMStack(
            input_dim=80,
            hidden_dims=[128, 128, 128, 128, 128],  # 5 layers
            rank=16,
            dropout=0.1
        )

    def test_deep_stack_forward(self, deep_stack):
        """Test deep stack forward pass."""
        batch_size = 2
        time_steps = 50
        x = torch.randn(batch_size, time_steps, 80)

        output, states, coord_info = deep_stack(x)

        assert output.shape == (batch_size, 128)
        assert len(states) == 5
        assert deep_stack.num_layers == 5

    def test_deep_stack_stability(self, deep_stack):
        """Test deep stack doesn't explode."""
        batch_size = 2
        time_steps = 100
        x = torch.randn(batch_size, time_steps, 80)

        output, states, coord_info = deep_stack(x)

        # Check for NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        for state in states:
            assert not torch.isnan(state.h).any()
            assert not torch.isinf(state.h).any()


class TestCoordinationVsUncoordinated:
    """Comparison tests."""

    def test_parameter_difference(self):
        """Test both have parameters."""
        coord = CoordinatedDREAMStack(
            input_dim=80,
            hidden_dims=[128, 128],
            rank=16
        )
        uncoord = UncoordinatedDREAMStack(
            input_dim=80,
            hidden_dims=[128, 128],
            rank=16
        )

        coord_params = coord.count_parameters()
        uncoord_params = uncoord.count_parameters()

        # Both should have reasonable parameter counts
        print(f"Coordinated: {coord_params:,}, Uncoordinated: {uncoord_params:,}")
        assert coord_params > 0
        assert uncoord_params > 0

    def test_output_shape_match(self):
        """Test both produce same output shape."""
        coord = CoordinatedDREAMStack(
            input_dim=80,
            hidden_dims=[128, 128],
            rank=16
        )
        uncoord = UncoordinatedDREAMStack(
            input_dim=80,
            hidden_dims=[128, 128],
            rank=16
        )

        x = torch.randn(2, 50, 80)

        coord_out, _, _ = coord(x)
        uncoord_out, _ = uncoord(x)

        assert coord_out.shape == uncoord_out.shape


class TestModulationFlow:
    """Tests for modulation flow through stack."""

    @pytest.fixture
    def stack(self):
        """Create stack for modulation testing."""
        return CoordinatedDREAMStack(
            input_dim=80,
            hidden_dims=[64, 64, 64],
            rank=8,
            dropout=0.0
        )

    def test_modulation_generated(self, stack):
        """Test that modulations are generated."""
        batch_size = 2
        time_steps = 50
        x = torch.randn(batch_size, time_steps, 80)

        output, states, coord_info = stack(x, return_all=True)

        # Should have modulations (some may be None for top layer)
        assert 'modulations' in coord_info
        assert len(coord_info['modulations']) > 0

    def test_modulation_shape(self, stack):
        """Test modulation shapes are correct."""
        batch_size = 2
        time_steps = 50
        x = torch.randn(batch_size, time_steps, 80)

        output, states, coord_info = stack(x, return_all=True)

        for i, mod in enumerate(coord_info['modulations']):
            if mod is not None:
                assert mod.shape == (batch_size, stack.hidden_dims[i])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
