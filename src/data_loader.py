"""
data_loader.py

Handles everything related to loading and preparing data:
  - Reading the raw .txt file from disk
  - Parsing it into numpy arrays
  - Splitting into train and test sets
  - Normalizing (standardizing) x and y so the network trains better

Each function does exactly one thing. They are meant to be composed together.
"""

import numpy as np


def read_file(filepath):
    """
    Read a raw text file and return its lines as a list of strings.

    We keep this separate so that parsing and I/O are not mixed together.
    If the file doesn't exist, Python raises a FileNotFoundError automatically.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    return lines


def parse_lines(lines):
    """
    Convert raw text lines into two numpy arrays: x (input) and y (target).

    Each line is expected to have two space-separated numbers, e.g.:
        -2 4.051976
    We skip blank lines so an empty trailing newline doesn't cause a crash.

    Returns:
        x  -- shape (n,)  the input values
        y  -- shape (n,)  the target values
    """
    x_values = []
    y_values = []

    for line in lines:
        # Skip lines that are empty or only whitespace
        stripped = line.strip()
        if not stripped:
            continue

        # Split the line on any whitespace and unpack the two numbers
        parts = stripped.split()
        x_values.append(float(parts[0]))
        y_values.append(float(parts[1]))

    # Convert Python lists to numpy arrays for fast math later
    x = np.array(x_values, dtype=np.float64)
    y = np.array(y_values, dtype=np.float64)

    return x, y


def compute_mean_std(array):
    """
    Compute the mean and standard deviation of a 1-D numpy array.

    We compute these separately from normalize() so that we can reuse the
    same statistics (from the training set) when normalizing the test set.

    Returns:
        mean  -- float, the average value
        std   -- float, the standard deviation (never zero; clamped to 1e-8)
    """
    mean = np.mean(array)
    std = np.std(array)

    # Protect against division by zero if all values happen to be identical
    if std == 0:
        std = 1e-8

    return mean, std


def normalize(array, mean, std):
    """
    Apply z-score (standard score) normalization: (value - mean) / std

    After this transform the output has approximately zero mean and unit variance.
    This helps the network converge faster during training.

    Args:
        array  -- 1-D numpy array of raw values
        mean   -- mean computed on the reference set (usually the training set)
        std    -- std  computed on the reference set (usually the training set)

    Returns:
        normalized array, same shape as input
    """
    return (array - mean) / std


def denormalize(array, mean, std):
    """
    Reverse z-score normalization: (value * std) + mean

    Useful when we want to display or evaluate predictions in the original scale.

    Args:
        array  -- 1-D numpy array of normalized values
        mean   -- the same mean that was used during normalization
        std    -- the same std  that was used during normalization

    Returns:
        array back in the original scale
    """
    return (array * std) + mean


def split_train_test(x, y, test_ratio=0.2, seed=42):
    """
    Randomly shuffle the data and split it into training and test sets.

    We shuffle first so the split is not biased by the order in the file
    (e.g. if the file was sorted by x, without shuffling the test set would
    only contain the largest x values, which is unfair).

    Args:
        x          -- 1-D numpy array of input values
        y          -- 1-D numpy array of target values (same length as x)
        test_ratio -- fraction of data to use for testing, e.g. 0.2 = 20%
        seed       -- random seed so the split is reproducible

    Returns:
        x_train, x_test, y_train, y_test  -- four 1-D numpy arrays
    """
    rng = np.random.default_rng(seed)

    # Create a shuffled list of indices, e.g. [3, 0, 7, 1, ...]
    indices = rng.permutation(len(x))

    # Calculate how many samples go into the test set
    n_test = int(len(x) * test_ratio)

    # The first n_test shuffled indices become the test set; the rest train
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def load_data(filepath, test_ratio=0.2, seed=42):
    """
    Full pipeline: read → parse → split → normalize → return.

    This is the main entry point. It ties all the smaller functions together
    and returns everything the network training code will need.

    The normalization statistics are computed only on the training set to
    avoid leaking information from the test set into the model.

    Args:
        filepath   -- path to a daneXX.txt file
        test_ratio -- fraction of data reserved for testing (default 20%)
        seed       -- random seed for reproducible train/test split

    Returns a dictionary with:
        x_train, x_test        -- normalized input arrays
        y_train, y_test        -- normalized target arrays
        x_mean, x_std          -- normalization stats for x
        y_mean, y_std          -- normalization stats for y (needed to denormalize predictions)
    """
    # --- 1. Read and parse the file ---
    lines = read_file(filepath)
    x, y = parse_lines(lines)

    # --- 2. Split into train and test BEFORE normalizing ---
    # (Important: we must not let test data influence the normalization stats)
    x_train_raw, x_test_raw, y_train_raw, y_test_raw = split_train_test(
        x, y, test_ratio=test_ratio, seed=seed
    )

    # --- 3. Compute normalization statistics from the TRAINING set only ---
    x_mean, x_std = compute_mean_std(x_train_raw)
    y_mean, y_std = compute_mean_std(y_train_raw)

    # --- 4. Apply normalization to both sets using the training statistics ---
    x_train = normalize(x_train_raw, x_mean, x_std)
    x_test = normalize(x_test_raw, x_mean, x_std)
    y_train = normalize(y_train_raw, y_mean, y_std)
    y_test = normalize(y_test_raw, y_mean, y_std)

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
