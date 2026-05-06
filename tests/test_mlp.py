"""
test_mlp.py

Unit tests for src/mlp.py and src/trainer.py

We test:
  - initialize_weights: correct shapes and reproducibility
  - MLP.forward: output shape, cache lengths, finite values
  - MLP.compute_loss: correctness, scalar output, non-negativity
  - MLP.backward: gradient shapes match weight shapes
  - MLP.update_weights: weights actually change
  - train_batch: loss decreases, history length is correct
  - train_online: loss decreases, one update per sample per epoch
  - get_activation_and_derivative: known values for tanh and sigmoid
  - prepare_inputs: 1-D arrays become (1, n) matrices
"""

import numpy as np
import pytest

from src.mlp import (MLP, initialize_weights, get_activation_and_derivative,
                     apply_relu, apply_relu_derivative)
from src.trainer import train_batch, train_online, train_online_one_epoch, prepare_inputs


# Tests for initialize_weights()


class TestInitializeWeights:

    def test_correct_number_of_matrices(self):
        """There should be one weight matrix per layer transition."""
        # [1, 10, 1] has 2 transitions: input→hidden and hidden→output
        weights, biases = initialize_weights([1, 10, 1])
        assert len(weights) == 2
        assert len(biases) == 2

    def test_correct_weight_shapes(self):
        """Each weight matrix should have shape (n_out, n_in)."""
        weights, _ = initialize_weights([1, 8, 4, 1])
        assert weights[0].shape == (8, 1)   # input(1) → hidden(8)
        assert weights[1].shape == (4, 8)   # hidden(8) → hidden(4)
        assert weights[2].shape == (1, 4)   # hidden(4) → output(1)

    def test_correct_bias_shapes(self):
        """Each bias should be a column vector of shape (n_out, 1)."""
        _, biases = initialize_weights([1, 8, 4, 1])
        assert biases[0].shape == (8, 1)
        assert biases[1].shape == (4, 1)
        assert biases[2].shape == (1, 1)

    def test_biases_initialized_to_zero(self):
        """Biases should all start at zero."""
        _, biases = initialize_weights([1, 10, 1])
        for b in biases:
            assert np.all(b == 0.0)

    def test_reproducible_with_same_seed(self):
        """The same seed must produce identical weight matrices."""
        w1, _ = initialize_weights([1, 10, 1], seed=7)
        w2, _ = initialize_weights([1, 10, 1], seed=7)
        assert np.array_equal(w1[0], w2[0])

    def test_different_seeds_produce_different_weights(self):
        """Different seeds should produce different starting weights."""
        w1, _ = initialize_weights([1, 10, 1], seed=0)
        w2, _ = initialize_weights([1, 10, 1], seed=1)
        assert not np.array_equal(w1[0], w2[0])


# Tests for MLP.forward()


class TestMLPForward:

    def setup_method(self):
        """Create a small network and a batch of 5 samples for every test."""
        self.mlp = MLP(layer_sizes=[1, 8, 1], activation="tanh")
        self.x = np.linspace(-1, 1, 5).reshape(1, 5)   # shape (1, 5)

    def test_output_shape(self):
        """forward() output should have shape (n_output, n_samples)."""
        y_pred = self.mlp.forward(self.x)
        # 1 output neuron, 5 samples → (1, 5)
        assert y_pred.shape == (1, 5)

    def test_z_cache_length(self):
        """z_cache should have one entry per layer (2 for a [1, 8, 1] network)."""
        self.mlp.forward(self.x)
        assert len(self.mlp.z_cache) == 2

    def test_a_cache_length(self):
        """a_cache contains the input plus one entry per layer = 3 for [1, 8, 1]."""
        self.mlp.forward(self.x)
        assert len(self.mlp.a_cache) == 3

    def test_output_is_finite(self):
        """The network output must not contain NaN or Inf."""
        y_pred = self.mlp.forward(self.x)
        assert np.all(np.isfinite(y_pred))

    def test_deeper_network_output_shape(self):
        """A deeper network [1, 16, 8, 4, 1] should still output shape (1, n)."""
        mlp = MLP(layer_sizes=[1, 16, 8, 4, 1], activation="tanh")
        x = np.linspace(-1, 1, 20).reshape(1, 20)
        y_pred = mlp.forward(x)
        assert y_pred.shape == (1, 20)


# Tests for MLP.compute_loss()


class TestComputeLoss:

    def setup_method(self):
        self.mlp = MLP(layer_sizes=[1, 8, 1])

    def test_loss_is_scalar(self):
        """MSE loss should be a single number, not an array."""
        y_pred = np.array([[1.0, 2.0, 3.0]])
        y_true = np.array([[1.5, 2.5, 3.5]])
        loss = self.mlp.compute_loss(y_pred, y_true)
        assert np.ndim(loss) == 0

    def test_loss_is_zero_for_perfect_predictions(self):
        """When predictions exactly match targets, MSE must be 0."""
        y = np.array([[1.0, -1.0, 0.5]])
        loss = self.mlp.compute_loss(y, y)
        assert np.isclose(loss, 0.0)

    def test_loss_is_non_negative(self):
        """MSE is a sum of squares so it can never be negative."""
        y_pred = np.array([[1.0, 2.0]])
        y_true = np.array([[3.0, 0.0]])
        loss = self.mlp.compute_loss(y_pred, y_true)
        assert loss >= 0.0

    def test_loss_value_is_correct(self):
        """Manually verify the MSE formula: mean((pred - true)^2)."""
        y_pred = np.array([[2.0, 4.0]])
        y_true = np.array([[1.0, 2.0]])
        # errors = [1, 2], squared = [1, 4], mean = 2.5
        loss = self.mlp.compute_loss(y_pred, y_true)
        assert np.isclose(loss, 2.5)

    def test_larger_error_gives_larger_loss(self):
        """A prediction further from the truth should produce a higher MSE."""
        y_true = np.array([[0.0]])
        loss_small = self.mlp.compute_loss(np.array([[0.1]]), y_true)
        loss_large = self.mlp.compute_loss(np.array([[10.0]]), y_true)
        assert loss_large > loss_small


# Tests for MLP.backward()


class TestMLPBackward:

    def setup_method(self):
        """Run a forward pass first so backward() has cached values to work with."""
        self.mlp = MLP(layer_sizes=[1, 8, 1], activation="tanh")
        x = np.linspace(-1, 1, 10).reshape(1, 10)
        self.mlp.forward(x)
        self.y_true = np.zeros((1, 10))

    def test_gradient_count_matches_weight_count(self):
        """backward() should return one gradient per weight matrix and bias."""
        dW_list, db_list = self.mlp.backward(self.y_true)
        assert len(dW_list) == len(self.mlp.weights)
        assert len(db_list) == len(self.mlp.biases)

    def test_weight_gradient_shapes(self):
        """Each dW should have the same shape as its corresponding weight matrix."""
        dW_list, _ = self.mlp.backward(self.y_true)
        for dW, W in zip(dW_list, self.mlp.weights):
            assert dW.shape == W.shape

    def test_bias_gradient_shapes(self):
        """Each db should have the same shape as its corresponding bias vector."""
        _, db_list = self.mlp.backward(self.y_true)
        for db, b in zip(db_list, self.mlp.biases):
            assert db.shape == b.shape

    def test_gradients_are_finite(self):
        """Gradients must not be NaN or Inf — that would break training."""
        dW_list, db_list = self.mlp.backward(self.y_true)
        for dW in dW_list:
            assert np.all(np.isfinite(dW))
        for db in db_list:
            assert np.all(np.isfinite(db))


# Tests for MLP.update_weights()


class TestUpdateWeights:

    def test_weights_change_after_update(self):
        """Weights should be different before and after update_weights() is called."""
        mlp = MLP(layer_sizes=[1, 8, 1])
        x = np.linspace(-1, 1, 10).reshape(1, 10)
        y = np.ones((1, 10))  # non-zero target so the gradient is non-zero

        original_W0 = mlp.weights[0].copy()

        mlp.forward(x)
        dW_list, db_list = mlp.backward(y)
        mlp.update_weights(dW_list, db_list, learning_rate=0.1)

        assert not np.array_equal(mlp.weights[0], original_W0)

    def test_zero_learning_rate_leaves_weights_unchanged(self):
        """With learning_rate=0 no update should happen."""
        mlp = MLP(layer_sizes=[1, 8, 1])
        x = np.linspace(-1, 1, 10).reshape(1, 10)
        y = np.ones((1, 10))

        original_W0 = mlp.weights[0].copy()

        mlp.forward(x)
        dW_list, db_list = mlp.backward(y)
        mlp.update_weights(dW_list, db_list, learning_rate=0.0)

        assert np.array_equal(mlp.weights[0], original_W0)


# Tests for train_batch()


class TestTrainBatch:

    def setup_method(self):
        """A simple y = x line is trivial to learn and good for sanity checks."""
        self.x_train = np.linspace(-1, 1, 40)
        self.y_train = self.x_train.copy()

    def test_loss_history_length(self):
        """train_batch() should return a list with exactly n_epochs entries."""
        mlp = MLP(layer_sizes=[1, 8, 1])
        history = train_batch(mlp, self.x_train, self.y_train, n_epochs=10, learning_rate=0.01)
        assert len(history) == 10

    def test_loss_decreases_over_training(self):
        """After enough epochs the final loss should be lower than the initial loss."""
        mlp = MLP(layer_sizes=[1, 16, 1], activation="tanh")
        history = train_batch(mlp, self.x_train, self.y_train, n_epochs=500, learning_rate=0.1)
        assert history[-1] < history[0]

    def test_all_losses_are_non_negative(self):
        """MSE can never be negative — every entry in loss_history must be >= 0."""
        mlp = MLP(layer_sizes=[1, 8, 1])
        history = train_batch(mlp, self.x_train, self.y_train, n_epochs=50, learning_rate=0.01)
        assert all(loss >= 0 for loss in history)

    def test_loss_history_values_are_floats(self):
        """Each element in loss_history should be a plain Python float."""
        mlp = MLP(layer_sizes=[1, 8, 1])
        history = train_batch(mlp, self.x_train, self.y_train, n_epochs=5, learning_rate=0.01)
        assert all(isinstance(loss, float) for loss in history)


# Tests for get_activation_and_derivative()


class TestActivations:

    def test_tanh_at_zero(self):
        """tanh(0) should be 0."""
        fn, _ = get_activation_and_derivative("tanh")
        assert np.isclose(fn(np.array([0.0])), 0.0)

    def test_tanh_derivative_at_zero(self):
        """tanh'(0) should be 1 — the maximum gradient of tanh."""
        _, deriv = get_activation_and_derivative("tanh")
        assert np.isclose(deriv(np.array([0.0])), 1.0)

    def test_sigmoid_at_zero(self):
        """sigmoid(0) should be exactly 0.5."""
        fn, _ = get_activation_and_derivative("sigmoid")
        assert np.isclose(fn(np.array([0.0])), 0.5)

    def test_sigmoid_derivative_at_zero(self):
        """sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25."""
        _, deriv = get_activation_and_derivative("sigmoid")
        assert np.isclose(deriv(np.array([0.0])), 0.25)

    def test_relu_at_positive(self):
        """ReLU should pass positive values through unchanged."""
        assert np.isclose(apply_relu(np.array([3.0])), 3.0)

    def test_relu_at_negative(self):
        """ReLU should clamp negative values to zero."""
        assert np.isclose(apply_relu(np.array([-5.0])), 0.0)

    def test_relu_at_zero(self):
        """ReLU(0) should be 0."""
        assert np.isclose(apply_relu(np.array([0.0])), 0.0)

    def test_relu_derivative_at_positive(self):
        """ReLU derivative is 1 for any positive input."""
        assert np.isclose(apply_relu_derivative(np.array([2.0])), 1.0)

    def test_relu_derivative_at_negative(self):
        """ReLU derivative is 0 for any negative input (gradient is blocked)."""
        assert np.isclose(apply_relu_derivative(np.array([-1.0])), 0.0)

    def test_relu_derivative_at_zero(self):
        """ReLU derivative at 0 is 0 by the standard subgradient convention."""
        assert np.isclose(apply_relu_derivative(np.array([0.0])), 0.0)

    def test_relu_registered_in_dispatcher(self):
        """get_activation_and_derivative('relu') should return the ReLU pair."""
        fn, deriv = get_activation_and_derivative("relu")
        # Spot-check: fn(2) = 2, deriv(2) = 1
        assert np.isclose(fn(np.array([2.0])), 2.0)
        assert np.isclose(deriv(np.array([2.0])), 1.0)

    def test_unknown_name_raises_value_error(self):
        """Requesting an unknown activation name should raise a ValueError."""
        with pytest.raises(ValueError):
            get_activation_and_derivative("swish_not_implemented")


# Tests for prepare_inputs()


class TestPrepareInputs:

    def test_shapes_become_2d(self):
        """1-D arrays of length n should become shape (1, n)."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        x_2d, y_2d = prepare_inputs(x, y)
        assert x_2d.shape == (1, 3)
        assert y_2d.shape == (1, 3)

    def test_values_are_preserved(self):
        """Reshaping should not change the actual values."""
        x = np.array([10.0, 20.0])
        y = np.array([-1.0, -2.0])
        x_2d, y_2d = prepare_inputs(x, y)
        assert np.array_equal(x_2d, [[10.0, 20.0]])
        assert np.array_equal(y_2d, [[-1.0, -2.0]])


# Tests for train_online() and train_online_one_epoch()


class TestTrainOnline:

    def setup_method(self):
        """Same simple y = x dataset used in batch training tests."""
        self.x_train = np.linspace(-1, 1, 40)
        self.y_train = self.x_train.copy()

    def test_loss_history_length(self):
        """train_online() should return a list with exactly n_epochs entries."""
        mlp = MLP(layer_sizes=[1, 8, 1])
        history = train_online(mlp, self.x_train, self.y_train, n_epochs=10, learning_rate=0.01)
        assert len(history) == 10

    def test_loss_decreases_over_training(self):
        """After enough epochs the final loss should be lower than the initial loss."""
        mlp = MLP(layer_sizes=[1, 16, 1], activation="tanh")
        history = train_online(mlp, self.x_train, self.y_train, n_epochs=300, learning_rate=0.05)
        assert history[-1] < history[0]

    def test_all_losses_are_non_negative(self):
        """Average MSE per epoch can never be negative."""
        mlp = MLP(layer_sizes=[1, 8, 1])
        history = train_online(mlp, self.x_train, self.y_train, n_epochs=20, learning_rate=0.01)
        assert all(loss >= 0 for loss in history)

    def test_loss_history_values_are_floats(self):
        """Each element in the loss history should be a plain Python float."""
        mlp = MLP(layer_sizes=[1, 8, 1])
        history = train_online(mlp, self.x_train, self.y_train, n_epochs=5, learning_rate=0.01)
        assert all(isinstance(loss, float) for loss in history)

    def test_same_seed_gives_same_history(self):
        """The same seed must produce the exact same loss curve every run."""
        mlp_a = MLP(layer_sizes=[1, 8, 1], seed=0)
        mlp_b = MLP(layer_sizes=[1, 8, 1], seed=0)
        history_a = train_online(mlp_a, self.x_train, self.y_train, n_epochs=10,
                                 learning_rate=0.01, seed=7)
        history_b = train_online(mlp_b, self.x_train, self.y_train, n_epochs=10,
                                 learning_rate=0.01, seed=7)
        assert history_a == history_b

    def test_weights_update_each_sample(self):
        """Weights should change after a single online epoch (n_samples updates)."""
        mlp = MLP(layer_sizes=[1, 8, 1])
        original_W0 = mlp.weights[0].copy()

        rng = np.random.default_rng(0)
        x_2d = self.x_train.reshape(1, -1)
        y_2d = self.y_train.reshape(1, -1)
        train_online_one_epoch(mlp, x_2d, y_2d, learning_rate=0.1, rng=rng)

        assert not np.array_equal(mlp.weights[0], original_W0)

    def test_online_and_batch_both_reduce_loss(self):
        """Both methods should reduce the loss — just via different update schedules."""
        mlp_batch = MLP(layer_sizes=[1, 16, 1], seed=0)
        mlp_online = MLP(layer_sizes=[1, 16, 1], seed=0)

        batch_history = train_batch(mlp_batch, self.x_train, self.y_train,
                                    n_epochs=200, learning_rate=0.05)
        online_history = train_online(mlp_online, self.x_train, self.y_train,
                                      n_epochs=200, learning_rate=0.005, seed=42)

        assert batch_history[-1] < batch_history[0]
        assert online_history[-1] < online_history[0]
