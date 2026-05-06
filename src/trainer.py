"""
trainer.py

Contains functions for training the MLP using batch gradient descent.

In batch training every weight update uses ALL training samples at once.
This produces stable, consistent gradients but can be slow on large datasets.
Step 4 of the task will introduce online (stochastic) training as a comparison.
"""


def prepare_inputs(x, y):
    """
    Reshape 1-D data arrays into 2-D matrices that the MLP expects.

    The MLP is written to work with shape (n_features, n_samples).
    Our data arrives as flat 1-D arrays of shape (n_samples,), so we add
    a leading dimension to make it (1, n_samples).

    Args:
        x -- 1-D numpy array of shape (n,)
        y -- 1-D numpy array of shape (n,)

    Returns:
        x_2d -- shape (1, n)
        y_2d -- shape (1, n)
    """
    x_2d = x.reshape(1, -1)
    y_2d = y.reshape(1, -1)
    return x_2d, y_2d


def train_one_epoch(mlp, x, y, learning_rate):
    """
    Run one full training pass: forward → loss → backward → update.

    This is the core training loop, kept as a single function so
    train_batch() can call it cleanly inside a for-loop.

    Args:
        mlp           -- an MLP instance (weights are modified in place)
        x             -- shape (1, n_samples), normalized inputs
        y             -- shape (1, n_samples), normalized targets
        learning_rate -- step size for gradient descent

    Returns:
        loss -- the MSE loss for this epoch (a single float)
    """
    # Forward pass: compute predictions
    y_pred = mlp.forward(x)

    # Measure how wrong the predictions are
    loss = mlp.compute_loss(y_pred, y)

    # Backward pass: compute gradients of the loss w.r.t. every weight
    dW_list, db_list = mlp.backward(y)

    # Update weights in the direction that reduces the loss
    mlp.update_weights(dW_list, db_list, learning_rate)

    return loss


def train_batch(mlp, x_train, y_train, n_epochs, learning_rate):
    """
    Train the MLP for a fixed number of epochs using batch gradient descent.

    Each epoch uses ALL training samples to compute a single gradient update.
    The loss after every epoch is recorded so we can plot the learning curve.

    Args:
        mlp           -- an MLP instance (modified in place)
        x_train       -- 1-D array of normalized training inputs
        y_train       -- 1-D array of normalized training targets
        n_epochs      -- how many full passes over the data to run
        learning_rate -- step size, e.g. 0.01

    Returns:
        loss_history -- list of floats, one MSE value per epoch
    """
    # Reshape to (1, n) so the MLP matrix operations work correctly
    x, y = prepare_inputs(x_train, y_train)

    loss_history = []

    for epoch in range(n_epochs):
        loss = train_one_epoch(mlp, x, y, learning_rate)
        loss_history.append(float(loss))

        # Print a progress update every 100 epochs so we can follow training
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1:>5}/{n_epochs}  |  MSE: {loss:.6f}")

    return loss_history
