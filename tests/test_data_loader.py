"""
test_data_loader.py

Unit tests for src/data_loader.py

We test each function independently so that if something breaks we know
exactly which piece caused the problem. Tests use small, hand-crafted
inputs so we can predict the expected output by hand.
"""

import os
import pytest
import numpy as np
import tempfile

from src.data_loader import (
    read_file,
    parse_lines,
    compute_mean_std,
    normalize,
    denormalize,
    split_train_test,
    load_data,
)


# Helpers

def make_temp_data_file(content):
    """
    Write a string to a temporary file and return its path.

    Using a temp file lets us test file I/O without touching the real data
    folder. The file is deleted manually after each test.
    """
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    tmp.write(content)
    tmp.close()
    return tmp.name


# Tests for read_file()

class TestReadFile:

    def test_returns_list_of_strings(self):
        """read_file() should return a list where each item is a string."""
        path = make_temp_data_file("-1 2.5\n0 3.0\n1 3.5\n")
        lines = read_file(path)
        os.unlink(path)

        assert isinstance(lines, list)
        assert all(isinstance(line, str) for line in lines)

    def test_correct_number_of_lines(self):
        """read_file() should return as many lines as are in the file."""
        path = make_temp_data_file("-1 2.5\n0 3.0\n1 3.5\n")
        lines = read_file(path)
        os.unlink(path)

        # The file has 3 lines so we expect 3 elements in the list
        assert len(lines) == 3

    def test_raises_on_missing_file(self):
        """read_file() should raise FileNotFoundError if the file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            read_file("this_file_does_not_exist.txt")


# Tests for parse_lines()

class TestParseLines:

    def test_returns_two_numpy_arrays(self):
        """parse_lines() should return exactly two numpy arrays."""
        lines = ["-1 2.5\n", "0 3.0\n", "1 3.5\n"]
        x, y = parse_lines(lines)

        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_correct_values_parsed(self):
        """parse_lines() should convert text numbers to correct float values."""
        lines = ["-2 4.05\n", "-1 3.43\n"]
        x, y = parse_lines(lines)

        # np.allclose checks floating-point equality with a small tolerance
        assert np.allclose(x, [-2.0, -1.0])
        assert np.allclose(y, [4.05, 3.43])

    def test_skips_blank_lines(self):
        """parse_lines() should ignore empty lines and not crash."""
        lines = ["-1 2.5\n", "\n", "1 3.5\n", "  \n"]
        x, y = parse_lines(lines)

        # Only 2 real data lines so we expect 2 values in each array
        assert len(x) == 2
        assert len(y) == 2

    def test_arrays_have_same_length(self):
        """x and y returned by parse_lines() must always be the same length."""
        lines = ["0 1.0\n", "1 2.0\n", "2 3.0\n"]
        x, y = parse_lines(lines)

        assert len(x) == len(y)


# Tests for compute_mean_std()

class TestComputeMeanStd:

    def test_correct_mean(self):
        """compute_mean_std() should return the correct mean."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, _ = compute_mean_std(arr)

        assert np.isclose(mean, 3.0)

    def test_correct_std(self):
        """compute_mean_std() should return the correct standard deviation."""
        arr = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        _, std = compute_mean_std(arr)

        # The population std of this array is 2.0
        assert np.isclose(std, 2.0)

    def test_std_never_zero(self):
        """compute_mean_std() must not return zero std (would cause division by zero)."""
        # An array with all identical values has std = 0 in theory
        arr = np.array([5.0, 5.0, 5.0])
        _, std = compute_mean_std(arr)

        # We clamp to 1e-8 to protect against division by zero later
        assert std > 0


# Tests for normalize()

class TestNormalize:

    def test_normalized_mean_is_zero(self):
        """After normalization with its own stats, the array mean should be ~0."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, std = compute_mean_std(arr)
        normalized = normalize(arr, mean, std)

        assert np.isclose(np.mean(normalized), 0.0, atol=1e-10)

    def test_normalized_std_is_one(self):
        """After normalization with its own stats, the array std should be ~1."""
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mean, std = compute_mean_std(arr)
        normalized = normalize(arr, mean, std)

        assert np.isclose(np.std(normalized), 1.0, atol=1e-10)

    def test_output_shape_unchanged(self):
        """normalize() should return an array of the same shape as the input."""
        arr = np.array([1.0, 2.0, 3.0])
        normalized = normalize(arr, mean=2.0, std=1.0)

        assert normalized.shape == arr.shape


# Tests for denormalize()

class TestDenormalize:

    def test_roundtrip_restores_original(self):
        """Normalizing then denormalizing should give back the original values."""
        arr = np.array([10.0, 20.0, 30.0])
        mean, std = compute_mean_std(arr)

        normalized = normalize(arr, mean, std)
        restored = denormalize(normalized, mean, std)

        # After the round-trip, values should match the original closely
        assert np.allclose(restored, arr)


# Tests for split_train_test()

class TestSplitTrainTest:

    def setup_method(self):
        """Create a small dataset shared across all split tests."""
        # 100 evenly spaced points — large enough to test the 80/20 ratio
        self.x = np.linspace(-5, 5, 100)
        self.y = self.x ** 2

    def test_total_samples_preserved(self):
        """train + test sizes must add up to the total number of samples."""
        x_train, x_test, y_train, y_test = split_train_test(self.x, self.y)

        assert len(x_train) + len(x_test) == len(self.x)

    def test_default_split_ratio_is_80_20(self):
        """With default test_ratio=0.2 and 100 samples, test should have 20 samples."""
        x_train, x_test, y_train, y_test = split_train_test(self.x, self.y)

        assert len(x_test) == 20
        assert len(x_train) == 80

    def test_split_is_reproducible_with_same_seed(self):
        """Calling split_train_test twice with the same seed must produce identical splits."""
        result_a = split_train_test(self.x, self.y, seed=0)
        result_b = split_train_test(self.x, self.y, seed=0)

        # Compare the training x arrays from both calls
        assert np.array_equal(result_a[0], result_b[0])

    def test_different_seeds_produce_different_splits(self):
        """Different seeds should (very likely) produce different orderings."""
        x_train_a, _, _, _ = split_train_test(self.x, self.y, seed=1)
        x_train_b, _, _, _ = split_train_test(self.x, self.y, seed=99)

        # With 100 points it is essentially impossible for two random shuffles to match
        assert not np.array_equal(x_train_a, x_train_b)

    def test_no_sample_appears_in_both_sets(self):
        """A data point must not appear in both train and test at the same time."""
        x_train, x_test, _, _ = split_train_test(self.x, self.y)

        # Convert to sets and check the intersection is empty
        train_set = set(x_train.tolist())
        test_set = set(x_test.tolist())

        assert len(train_set & test_set) == 0


# Tests for load_data() — the full pipeline

class TestLoadData:

    def setup_method(self):
        """Create a temp file that looks like a real daneXX.txt file."""
        rng = np.random.default_rng(0)
        xs = np.linspace(-2, 2, 50)
        ys = np.sin(xs) + rng.normal(0, 0.1, size=50)

        # Build the file content: two columns separated by a space
        content = "\n".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
        self.path = make_temp_data_file(content)

    def teardown_method(self):
        """Remove the temp file after each test."""
        os.unlink(self.path)

    def test_returns_dict_with_all_keys(self):
        """load_data() must return a dict containing all expected keys."""
        result = load_data(self.path)
        expected_keys = {"x_train", "x_test", "y_train", "y_test",
                         "x_mean", "x_std", "y_mean", "y_std"}

        assert expected_keys == set(result.keys())

    def test_train_and_test_sizes_add_up(self):
        """x_train + x_test must together contain all 50 samples."""
        result = load_data(self.path)

        assert len(result["x_train"]) + len(result["x_test"]) == 50

    def test_training_x_mean_is_near_zero(self):
        """Normalized x_train should have a mean very close to 0."""
        result = load_data(self.path)

        assert np.isclose(np.mean(result["x_train"]), 0.0, atol=1e-10)

    def test_training_x_std_is_near_one(self):
        """Normalized x_train should have a std very close to 1."""
        result = load_data(self.path)

        assert np.isclose(np.std(result["x_train"]), 1.0, atol=1e-10)

    def test_normalization_stats_are_scalars(self):
        """x_mean, x_std, y_mean, y_std should all be scalars, not arrays."""
        result = load_data(self.path)

        for key in ["x_mean", "x_std", "y_mean", "y_std"]:
            # np.ndim(scalar) == 0, meaning it's not an array
            assert np.ndim(result[key]) == 0, f"{key} should be a scalar, not an array"

    def test_custom_test_ratio(self):
        """load_data() should respect a custom test_ratio argument."""
        result = load_data(self.path, test_ratio=0.3)

        # 30% of 50 = 15 test samples
        assert len(result["x_test"]) == 15
