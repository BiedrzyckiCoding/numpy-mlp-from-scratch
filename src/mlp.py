"""
mlp.py

Implements the MLP (Multi-Layer Perceptron) neural network from scratch using numpy.

The network is designed for 1D regression: one input, one output.
It supports configurable hidden layer sizes and activation functions (tanh or sigmoid).
"""

import numpy as np


# Activation functions and their derivatives


def apply_tanh(z):
    """Apply the tanh activation function element-wise."""
    return np.tanh(z)


def apply_tanh_derivative(z):
    """Derivative of tanh: 1 - tanh(z)^2. Used in backpropagation."""
    return 1.0 - np.tanh(z) ** 2


def apply_sigmoid(z):
    """Apply the sigmoid activation function element-wise: 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-z))


def apply_sigmoid_derivative(z):
    """Derivative of sigmoid: sigmoid(z) * (1 - sigmoid(z)). Used in backpropagation."""
    s = apply_sigmoid(z)
    return s * (1.0 - s)


def apply_relu(z):
    """
    Apply the ReLU (Rectified Linear Unit) activation element-wise.

    ReLU(z) = max(0, z)

    Negative values become 0; positive values pass through unchanged.
    This is computationally cheap and avoids the vanishing-gradient problem
    that tanh and sigmoid can suffer from in deep networks.
    """
    return np.maximum(0.0, z)


def apply_relu_derivative(z):
    """
    Derivative of ReLU element-wise.

    ReLU'(z) = 1 if z > 0, else 0

    At exactly z=0 the derivative is technically undefined, but we follow
    the standard convention of returning 0 there (subgradient).
    """
    return (z > 0).astype(float)


def get_activation_and_derivative(name):
    """
    Return the activation function and its derivative as a pair.

    We return both together so the MLP doesn't need if/else chains in
    forward() and backward() — it just calls whatever functions it holds.

    Args:
        name -- 'tanh', 'sigmoid', or 'relu'

    Returns:
        (activation_fn, derivative_fn) -- two callable functions
    """
    if name == "tanh":
        return apply_tanh, apply_tanh_derivative
    elif name == "sigmoid":
        return apply_sigmoid, apply_sigmoid_derivative
    elif name == "relu":
        return apply_relu, apply_relu_derivative
    else:
        raise ValueError(f"Unknown activation: '{name}'. Choose 'tanh', 'sigmoid', or 'relu'.")


# Weight initialization


def initialize_weights(layer_sizes, seed=42):
    """
    Create weight matrices and bias vectors for each layer transition.

    We use Xavier uniform initialization, which sets weights to small random
    values scaled by the size of the surrounding layers. This prevents the
    signal from vanishing or exploding at the start of training.

    Args:
        layer_sizes -- list of ints, e.g. [1, 10, 1] means input=1, hidden=10, output=1
        seed        -- random seed for reproducibility

    Returns:
        weights -- list of 2D arrays, one per layer transition; shape (n_out, n_in)
        biases  -- list of column vectors; shape (n_out, 1), initialized to zero
    """
    rng = np.random.default_rng(seed)
    weights = []
    biases = []

    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i + 1]

        # Xavier uniform: scale limits by sqrt(6 / (n_in + n_out))
        limit = np.sqrt(6.0 / (n_in + n_out))
        W = rng.uniform(-limit, limit, size=(n_out, n_in))

        # Biases start at zero — weights carry all the initial asymmetry
        b = np.zeros((n_out, 1))

        weights.append(W)
        biases.append(b)

    return weights, biases


# The MLP class


class MLP:
    """
    A simple Multi-Layer Perceptron for 1D regression.

    Example usage:
        mlp    = MLP(layer_sizes=[1, 10, 1], activation='tanh')
        y_pred = mlp.forward(x)                          # x shape: (1, n_samples)
        loss   = mlp.compute_loss(y_pred, y_true)
        dW, db = mlp.backward(y_true)
        mlp.update_weights(dW, db, learning_rate=0.01)
    """

    def __init__(self, layer_sizes, activation="tanh", seed=42):
        """
        Set up the network structure.

        Args:
            layer_sizes -- e.g. [1, 10, 10, 1] for 2 hidden layers of 10 neurons each
            activation  -- activation for hidden layers: 'tanh' or 'sigmoid'
            seed        -- for reproducible weight initialization
        """
        self.layer_sizes = layer_sizes
        self.activation_name = activation

        # Store the activation function and its derivative as callable attributes
        self.activation_fn, self.activation_deriv = get_activation_and_derivative(activation)

        # Initialize all weights and biases
        self.weights, self.biases = initialize_weights(layer_sizes, seed=seed)

        # These lists are populated during forward() and read during backward()
        self.z_cache = []  # pre-activation values at each layer
        self.a_cache = []  # post-activation values (a_cache[0] = network input)

    def forward(self, x):
        """
        Pass the input through every layer and return the final prediction.

        The result is stored in self.a_cache[-1] and also returned directly.
        Intermediate values are saved in self.z_cache and self.a_cache so
        backward() can use them without recomputing anything.

        Args:
            x -- numpy array of shape (n_input, n_samples)

        Returns:
            output -- numpy array of shape (n_output, n_samples)
        """
        # Clear caches from any previous call
        self.z_cache = []
        self.a_cache = [x]  # a_cache[0] is the raw input

        current = x

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            # Linear combination: z = W @ a_prev + b
            z = W @ current + b
            self.z_cache.append(z)

            is_output_layer = (i == len(self.weights) - 1)
            if is_output_layer:
                # Output layer has no activation — we want an unbounded real number
                current = z
            else:
                # Hidden layers apply the chosen non-linear activation
                current = self.activation_fn(z)

            self.a_cache.append(current)

        return current

    def compute_loss(self, y_pred, y_true):
        """
        Compute Mean Squared Error (MSE) between predictions and ground truth.

        MSE = (1/n) * sum((y_pred - y_true)^2)

        A perfect model would have MSE = 0.

        Args:
            y_pred -- shape (1, n_samples), network output
            y_true -- shape (1, n_samples), ground truth targets

        Returns:
            loss -- a single scalar float
        """
        n_samples = y_true.shape[1]
        squared_errors = (y_pred - y_true) ** 2
        loss = np.sum(squared_errors) / n_samples
        return loss

    def backward(self, y_true):
        """
        Compute gradients of the MSE loss w.r.t. every weight and bias.

        This is the backpropagation algorithm. It applies the chain rule going
        backwards from the output layer to the first hidden layer.

        Must be called AFTER forward() — it reads from self.z_cache and self.a_cache.

        Args:
            y_true -- shape (1, n_samples), the true target values

        Returns:
            dW_list -- list of gradient arrays, same shapes as self.weights
            db_list -- list of gradient arrays, same shapes as self.biases
        """
        n_samples = y_true.shape[1]
        n_layers = len(self.weights)

        # Pre-allocate lists so we can fill them in reverse order
        dW_list = [None] * n_layers
        db_list = [None] * n_layers

        # The output layer's delta is just the MSE gradient: (y_pred - y_true) / n
        # a_cache[-1] holds the network's final output (y_pred)
        delta = (self.a_cache[-1] - y_true) / n_samples

        # Walk backwards through all layers
        for i in reversed(range(n_layers)):
            # Weight gradient: outer product of delta and the previous activation
            dW_list[i] = delta @ self.a_cache[i].T

            # Bias gradient: sum delta across the sample axis
            db_list[i] = np.sum(delta, axis=1, keepdims=True)

            if i > 0:
                # Pass the error signal back through the weight matrix ...
                delta = self.weights[i].T @ delta
                # ... and through the activation function's derivative
                delta = delta * self.activation_deriv(self.z_cache[i - 1])

        return dW_list, db_list

    def update_weights(self, dW_list, db_list, learning_rate):
        """
        Apply one step of gradient descent to all weights and biases.

        The update rule is: w = w - learning_rate * gradient

        A large learning rate takes bigger steps (faster, but may overshoot).
        A small learning rate takes smaller steps (safer, but slower).

        Args:
            dW_list       -- weight gradients from backward()
            db_list       -- bias gradients from backward()
            learning_rate -- scalar step size, e.g. 0.01
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW_list[i]
            self.biases[i] -= learning_rate * db_list[i]
