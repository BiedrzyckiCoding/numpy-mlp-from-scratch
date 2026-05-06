"""
evaluator.py

Functions for evaluating how well the trained network fits the data.

Covers three concerns:
  - Metrics: MSE and R² in the original (denormalized) scale
  - Diagnosis: underfitting / good fit / overfitting
  - Visualization: fit plot and loss curve
"""

import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import normalize, denormalize


# Metrics


def compute_mse(y_pred, y_true):
    """
    Compute Mean Squared Error between predictions and ground truth.

    Both arrays must be in the SAME scale (either both normalized or both original).
    Lower is better; 0.0 means perfect predictions.

    Args:
        y_pred -- 1-D numpy array of predicted values
        y_true -- 1-D numpy array of true values

    Returns:
        mse -- a single float
    """
    squared_errors = (y_pred - y_true) ** 2
    return float(np.mean(squared_errors))


def compute_r2(y_pred, y_true):
    """
    Compute the R² (coefficient of determination) score.

    R² tells us how much of the variance in the data the model explains:
      1.0  → perfect fit
      0.0  → model is as good as just predicting the mean
      < 0  → model is worse than predicting the mean

    Formula: R² = 1 - (SS_residual / SS_total)
      SS_residual = sum of squared prediction errors
      SS_total    = sum of squared deviations from the mean

    Args:
        y_pred -- 1-D numpy array of predicted values
        y_true -- 1-D numpy array of true values

    Returns:
        r2 -- a single float
    """
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)

    # Protect against division by zero if all true values are identical
    if ss_total == 0:
        return 1.0 if ss_residual == 0 else 0.0

    return float(1.0 - ss_residual / ss_total)


# Prediction in original scale


def predict(mlp, x_raw, x_mean, x_std, y_mean, y_std):
    """
    Run the network on raw (unnormalized) x values and return predictions
    in the original y scale.

    The network was trained on normalized data, so we must:
      1. Normalize x using the training statistics
      2. Run the forward pass
      3. Denormalize the output back to the original scale

    Args:
        mlp    -- trained MLP instance
        x_raw  -- 1-D numpy array of unnormalized input values
        x_mean -- mean used to normalize x during training
        x_std  -- std  used to normalize x during training
        y_mean -- mean used to normalize y during training
        y_std  -- std  used to normalize y during training

    Returns:
        y_pred_original -- 1-D numpy array of predictions in original scale
    """
    # Normalize the input the same way the training data was normalized
    x_norm = normalize(x_raw, x_mean, x_std)

    # The MLP expects shape (1, n_samples), so add a leading dimension
    x_input = x_norm.reshape(1, -1)

    # Run the forward pass — output has shape (1, n_samples)
    y_norm = mlp.forward(x_input)

    # Flatten back to 1-D and convert from normalized to original scale
    y_pred_original = denormalize(y_norm.flatten(), y_mean, y_std)

    return y_pred_original


# Fit diagnosis


def assess_fit(train_mse, test_mse, overfit_ratio=2.5, underfit_threshold=0.5):
    """
    Classify the model's fit as underfitting, good fit, or overfitting.

    Rules:
      - Underfitting: train MSE is high — the model hasn't learned the data at all
      - Overfitting:  test MSE is much larger than train MSE — the model memorized
                      the training set but doesn't generalize
      - Good fit:     train and test MSEs are both low and close to each other

    Args:
        train_mse          -- MSE on the training set (in normalized scale)
        test_mse           -- MSE on the test set (in normalized scale)
        overfit_ratio      -- test_mse / train_mse above this → overfitting
        underfit_threshold -- train_mse above this → underfitting

    Returns:
        diagnosis -- one of: 'underfitting', 'overfitting', 'good fit'
        message   -- a human-readable explanation string
    """
    # Check for underfitting first: if train error itself is large, the model
    # hasn't even learned the training data — adding more data won't help here
    if train_mse > underfit_threshold:
        return "underfitting", (
            f"Train MSE ({train_mse:.4f}) is high. "
            "The model is too simple or needs more training."
        )

    # Check for overfitting: test error is disproportionately larger than train error
    if test_mse > overfit_ratio * train_mse:
        return "overfitting", (
            f"Test MSE ({test_mse:.4f}) >> Train MSE ({train_mse:.4f}). "
            "The model memorized the training data but doesn't generalize."
        )

    # Both errors are low and close — the model generalizes well
    return "good fit", (
        f"Train MSE ({train_mse:.4f}), Test MSE ({test_mse:.4f}). "
        "The model generalizes well."
    )


# Visualization


def plot_fit(x_train, y_train, x_test, y_test, x_curve, y_curve, title="Model fit"):
    """
    Plot training points, test points, and the model's predicted curve on one chart.

    This gives a visual sense of how well the network approximates the function:
      - Blue dots  → training data
      - Red dots   → test data
      - Black line → model predictions across the full x range

    Args:
        x_train -- 1-D array, training input values (original scale)
        y_train -- 1-D array, training target values (original scale)
        x_test  -- 1-D array, test input values (original scale)
        y_test  -- 1-D array, test target values (original scale)
        x_curve -- 1-D array, evenly spaced x values for drawing the model curve
        y_curve -- 1-D array, model predictions at x_curve (original scale)
        title   -- string title for the chart

    Returns:
        fig -- the matplotlib Figure object (useful for saving or testing)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot training and test points as scatter dots
    ax.scatter(x_train, y_train, s=20, alpha=0.7, label="Train data", color="steelblue")
    ax.scatter(x_test, y_test, s=20, alpha=0.7, label="Test data", color="tomato")

    # Sort x_curve so the line is drawn left-to-right without zigzagging
    sort_idx = np.argsort(x_curve)
    ax.plot(x_curve[sort_idx], y_curve[sort_idx], color="black", linewidth=2, label="Model")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.tight_layout()

    return fig


def plot_loss_curve(loss_history, title="Training loss"):
    """
    Plot MSE loss vs. epoch number to visualize how training progressed.

    A healthy training curve should decrease smoothly and then flatten out.
    If it goes up again, the model may be diverging (learning rate too high).

    Args:
        loss_history -- list of float MSE values, one per epoch
        title        -- string title for the chart

    Returns:
        fig -- the matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # epoch numbers start at 1 for readability on the x-axis
    epochs = range(1, len(loss_history) + 1)

    ax.plot(epochs, loss_history, color="steelblue", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")

    # Log scale on y makes it easier to see progress when loss drops fast early on
    ax.set_yscale("log")

    plt.tight_layout()
    return fig
