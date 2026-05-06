"""
test_evaluator.py

Unit tests for src/evaluator.py

We test:
  - compute_mse: correct value, non-negative, zero for perfect predictions
  - compute_r2:  correct value, 1.0 for perfect fit, ≤ 1 always
  - predict:     output shape, values are in original scale
  - assess_fit:  correct diagnosis for each of the three cases
  - plot_fit and plot_loss_curve: return a Figure, don't crash
"""

import numpy as np
import pytest
import matplotlib
# Use the non-interactive Agg backend so plots render without a display
# This is important for running tests in CI (no monitor attached)
matplotlib.use("Agg")

from src.evaluator import compute_mse, compute_r2, predict, assess_fit, plot_fit, plot_loss_curve
from src.mlp import MLP
from src.trainer import train_batch


# Tests for compute_mse()


class TestComputeMse:

    def test_zero_for_perfect_predictions(self):
        """MSE should be 0 when predictions exactly match the targets."""
        y = np.array([1.0, 2.0, 3.0])
        assert np.isclose(compute_mse(y, y), 0.0)

    def test_correct_value(self):
        """Manually verify the MSE formula: mean((pred - true)^2)."""
        y_pred = np.array([2.0, 4.0])
        y_true = np.array([1.0, 2.0])
        # errors = [1, 2], squared = [1, 4], mean = 2.5
        assert np.isclose(compute_mse(y_pred, y_true), 2.5)

    def test_is_non_negative(self):
        """MSE is a sum of squares so it can never be negative."""
        y_pred = np.array([10.0, -5.0, 3.0])
        y_true = np.array([0.0, 0.0, 0.0])
        assert compute_mse(y_pred, y_true) >= 0.0

    def test_returns_float(self):
        """compute_mse() should return a plain Python float."""
        y = np.array([1.0, 2.0])
        result = compute_mse(y, y)
        assert isinstance(result, float)

    def test_symmetric(self):
        """MSE(a, b) should equal MSE(b, a) because errors are squared."""
        y_pred = np.array([1.0, 3.0])
        y_true = np.array([2.0, 0.0])
        assert np.isclose(compute_mse(y_pred, y_true), compute_mse(y_true, y_pred))


# Tests for compute_r2()


class TestComputeR2:

    def test_one_for_perfect_predictions(self):
        """R² should be exactly 1.0 when predictions are perfect."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isclose(compute_r2(y, y), 1.0)

    def test_zero_for_mean_predictor(self):
        """R² should be 0 when the model always predicts the mean of y_true."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Predicting the mean every time gives R² = 0 by definition
        y_pred = np.full_like(y_true, np.mean(y_true))
        assert np.isclose(compute_r2(y_pred, y_true), 0.0)

    def test_negative_for_terrible_predictions(self):
        """R² can be negative when the model is worse than predicting the mean."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])  # wildly wrong
        assert compute_r2(y_pred, y_true) < 0.0

    def test_returns_float(self):
        """compute_r2() should return a plain Python float."""
        y = np.array([1.0, 2.0, 3.0])
        assert isinstance(compute_r2(y, y), float)

    def test_constant_y_true_all_same(self):
        """If all true values are identical, R² should not crash."""
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        # Perfect predictions on a constant target → 1.0
        assert np.isclose(compute_r2(y_pred, y_true), 1.0)


# Tests for predict()


class TestPredict:

    def setup_method(self):
        """Train a tiny network so we have something to call predict() on."""
        # Simple dataset: y = x
        x_train = np.linspace(-1, 1, 40)
        y_train = x_train.copy()

        # These are the normalization stats (for identity data, mean=0, std≈0.59)
        self.x_mean = float(np.mean(x_train))
        self.x_std = float(np.std(x_train))
        self.y_mean = float(np.mean(y_train))
        self.y_std = float(np.std(y_train))

        # Normalize and train a small network
        x_norm = (x_train - self.x_mean) / self.x_std
        y_norm = (y_train - self.y_mean) / self.y_std

        self.mlp = MLP(layer_sizes=[1, 8, 1], activation="tanh")
        train_batch(self.mlp, x_norm, y_norm, n_epochs=1, learning_rate=0.01)

        # A few raw x values to run predictions on
        self.x_raw = np.array([-1.0, 0.0, 1.0])

    def test_output_length_matches_input(self):
        """predict() should return one value for each input point."""
        y_pred = predict(self.mlp, self.x_raw, self.x_mean, self.x_std, self.y_mean, self.y_std)
        assert len(y_pred) == len(self.x_raw)

    def test_output_is_1d(self):
        """predict() should return a 1-D array, not a 2-D matrix."""
        y_pred = predict(self.mlp, self.x_raw, self.x_mean, self.x_std, self.y_mean, self.y_std)
        assert y_pred.ndim == 1

    def test_output_is_finite(self):
        """Predictions must not contain NaN or Inf."""
        y_pred = predict(self.mlp, self.x_raw, self.x_mean, self.x_std, self.y_mean, self.y_std)
        assert np.all(np.isfinite(y_pred))


# Tests for assess_fit()


class TestAssessFit:

    def test_detects_underfitting(self):
        """High train MSE should be classified as underfitting."""
        diagnosis, _ = assess_fit(train_mse=0.9, test_mse=0.95)
        assert diagnosis == "underfitting"

    def test_detects_overfitting(self):
        """Test MSE much larger than train MSE should be classified as overfitting."""
        # test/train ratio = 10/0.01 = 1000 — clearly overfitting
        diagnosis, _ = assess_fit(train_mse=0.01, test_mse=10.0)
        assert diagnosis == "overfitting"

    def test_detects_good_fit(self):
        """Low train MSE and test MSE close together should be a good fit."""
        diagnosis, _ = assess_fit(train_mse=0.02, test_mse=0.03)
        assert diagnosis == "good fit"

    def test_returns_a_message_string(self):
        """assess_fit() should always return a non-empty explanation string."""
        _, message = assess_fit(train_mse=0.02, test_mse=0.03)
        assert isinstance(message, str)
        assert len(message) > 0

    def test_underfitting_takes_priority_over_overfitting(self):
        """If train MSE is already high, we call it underfitting regardless of test MSE."""
        # Even though test_mse >> train_mse, train_mse is too high to call it overfitting
        diagnosis, _ = assess_fit(train_mse=0.8, test_mse=5.0)
        assert diagnosis == "underfitting"


# Tests for plot_fit() and plot_loss_curve()


class TestPlots:

    def setup_method(self):
        """Create simple arrays used by both plot tests."""
        self.x = np.linspace(-2, 2, 20)
        self.y = np.sin(self.x)

    def test_plot_fit_returns_figure(self):
        """plot_fit() should return a matplotlib Figure without crashing."""
        import matplotlib.figure
        fig = plot_fit(
            x_train=self.x[:15], y_train=self.y[:15],
            x_test=self.x[15:], y_test=self.y[15:],
            x_curve=self.x, y_curve=self.y,
            title="Test fit plot",
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_loss_curve_returns_figure(self):
        """plot_loss_curve() should return a matplotlib Figure without crashing."""
        import matplotlib.figure
        loss_history = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
        fig = plot_loss_curve(loss_history, title="Test loss curve")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_loss_curve_single_epoch(self):
        """plot_loss_curve() should not crash when given just one data point."""
        import matplotlib.figure
        fig = plot_loss_curve([0.5])
        assert isinstance(fig, matplotlib.figure.Figure)
