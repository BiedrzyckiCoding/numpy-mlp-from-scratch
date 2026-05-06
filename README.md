# NumPy MLP from Scratch

A multi-layer perceptron (MLP) neural network built entirely with **NumPy** — no PyTorch, no TensorFlow, no autograd. Every forward pass, backpropagation step, and gradient update is implemented by hand.

The project ships with a **Flask web UI** that walks through all five task steps interactively, and a **CI/CD pipeline** (GitHub Actions) that runs lint and 93 unit tests on every push.

---

## Table of contents

1. [Project structure](#project-structure)
2. [Setup](#setup)
3. [Running the Flask app](#running-the-flask-app)
4. [Running the tests](#running-the-tests)
5. [Step 1 — Data loading & splitting](#step-1--data-loading--splitting)
6. [Step 2 — Neural network & batch training](#step-2--neural-network--batch-training)
7. [Step 3 — Evaluation](#step-3--evaluation)
8. [Step 4 — Online (stochastic) training](#step-4--online-stochastic-training)
9. [Step 5 — ReLU activation](#step-5--relu-activation)
10. [CI/CD pipeline](#cicd-pipeline)
11. [Test suite overview](#test-suite-overview)

---

## Project structure

```
numpy-mlp-from-scratch/
│
├── data/                      # 16 regression datasets (dane1.txt … dane16.txt)
│
├── src/
│   ├── data_loader.py         # Step 1 — read, parse, split, normalize
│   ├── mlp.py                 # Step 2 — MLP class, activations, Xavier init
│   ├── trainer.py             # Steps 2 & 4 — batch and online training loops
│   └── evaluator.py           # Step 3 — MSE, R², fit plot, loss curve
│
├── tests/
│   ├── conftest.py            # Sets matplotlib Agg backend before all tests
│   ├── test_data_loader.py    # 25 tests for src/data_loader.py
│   ├── test_mlp.py            # 47 tests for src/mlp.py and src/trainer.py
│   └── test_evaluator.py      # 21 tests for src/evaluator.py
│
├── templates/                 # Jinja2 HTML templates (Flask)
│   ├── base.html              # Sidebar layout shell
│   ├── index.html             # Landing page
│   ├── data.html              # Step 1 UI
│   ├── train.html             # Step 2 UI
│   ├── evaluate.html          # Step 3 UI
│   ├── compare.html           # Step 4 UI
│   └── relu.html              # Step 5 UI
│
├── static/css/style.css       # All UI styles
├── app.py                     # Flask application entry point
├── requirements.txt
└── .github/workflows/ci.yml   # GitHub Actions CI/CD pipeline
```

---

## Setup

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/BiedrzyckiCoding/numpy-mlp-from-scratch.git
cd numpy-mlp-from-scratch

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` pins:

```
numpy==1.26.4
matplotlib==3.9.0
flask==3.0.3
pytest==8.2.0
pytest-cov==5.0.0
flake8==7.0.0
```

---

## Running the Flask app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser. The sidebar links to each of the five steps. Training results are held in memory so the Evaluation page can reuse the model trained on the Train page without retraining.

---

## Running the tests

```bash
# Run all 93 tests
pytest tests/ -v

# Run with coverage report in the terminal
pytest tests/ --cov=src --cov-report=term-missing

# Run a single test file
pytest tests/test_data_loader.py -v
```

---

## Step 1 — Data loading & splitting

**File:** `src/data_loader.py`

Each dataset is a plain text file with two space-separated columns:

```
-2 4.051976
-1.9 3.426599
-1.8 3.403733
...
```

The loader is split into small, single-purpose functions that are composed together.

### Reading and parsing

```python
from src.data_loader import read_file, parse_lines

lines = read_file("data/dane1.txt")   # returns a list of strings
x, y  = parse_lines(lines)            # returns two 1-D numpy arrays
```

`parse_lines` skips blank lines and converts each row to `float64`:

```python
def parse_lines(lines):
    x_values, y_values = [], []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        x_values.append(float(parts[0]))
        y_values.append(float(parts[1]))
    return np.array(x_values), np.array(y_values)
```

### Train / test split

```python
from src.data_loader import split_train_test

x_train, x_test, y_train, y_test = split_train_test(x, y, test_ratio=0.2, seed=42)
```

Data is shuffled before splitting so the test set is not biased by the file's ordering (e.g. sorted x values).

### Z-score normalisation

```python
from src.data_loader import compute_mean_std, normalize, denormalize

# Compute stats on the TRAINING set only — never the test set
x_mean, x_std = compute_mean_std(x_train)
y_mean, y_std = compute_mean_std(y_train)

# Apply to both sets using training statistics
x_train_norm = normalize(x_train, x_mean, x_std)   # (x - mean) / std
x_test_norm  = normalize(x_test,  x_mean, x_std)

# Reverse the transform later for plotting
x_original = denormalize(x_train_norm, x_mean, x_std)
```

Using training statistics on the test set is critical — fitting the scaler to the test data would leak information and inflate performance metrics.

### Full pipeline in one call

```python
from src.data_loader import load_data

result = load_data("data/dane1.txt", test_ratio=0.2, seed=42)
# result is a dict with keys:
#   x_train, x_test, y_train, y_test  — normalized arrays
#   x_mean, x_std, y_mean, y_std      — stats needed to denormalize later
```

---

## Step 2 — Neural network & batch training

**Files:** `src/mlp.py`, `src/trainer.py`

### Activation functions

Three activations are implemented. Each is a plain function so the MLP can hold them as callable attributes.

```python
# tanh — output in [-1, 1], smooth gradient everywhere
def apply_tanh(z):
    return np.tanh(z)

def apply_tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2

# sigmoid — output in [0, 1]
def apply_sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def apply_sigmoid_derivative(z):
    s = apply_sigmoid(z)
    return s * (1.0 - s)

# ReLU — output in [0, ∞), no vanishing gradient for positive inputs
def apply_relu(z):
    return np.maximum(0.0, z)

def apply_relu_derivative(z):
    return (z > 0).astype(float)
```

To fetch an activation pair by name:

```python
from src.mlp import get_activation_and_derivative

fn, deriv = get_activation_and_derivative("tanh")
```

### Weight initialisation — Xavier uniform

```python
def initialize_weights(layer_sizes, seed=42):
    rng = np.random.default_rng(seed)
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        n_in  = layer_sizes[i]
        n_out = layer_sizes[i + 1]
        limit = np.sqrt(6.0 / (n_in + n_out))
        W = rng.uniform(-limit, limit, size=(n_out, n_in))
        b = np.zeros((n_out, 1))   # biases start at zero
        weights.append(W)
        biases.append(b)
    return weights, biases
```

Xavier uniform scales the initial weights by the layer sizes. This prevents the gradient signal from vanishing or exploding before training even starts.

### MLP class — forward pass

```python
from src.mlp import MLP

mlp = MLP(layer_sizes=[1, 16, 16, 1], activation="tanh", seed=42)

# x must be shape (n_features, n_samples) — here (1, n)
x = x_train.reshape(1, -1)
y_pred = mlp.forward(x)    # shape (1, n)
```

Inside `forward`, each layer applies a linear transform then an activation. The output layer is **linear** (no activation) so it can predict unbounded real numbers:

```python
for i, (W, b) in enumerate(zip(self.weights, self.biases)):
    z = W @ current + b          # linear step
    is_output_layer = (i == len(self.weights) - 1)
    if is_output_layer:
        current = z              # no activation on the output
    else:
        current = self.activation_fn(z)
    self.z_cache.append(z)
    self.a_cache.append(current)
```

### MLP class — MSE loss

```python
loss = mlp.compute_loss(y_pred, y_true)
# MSE = (1/n) * sum((y_pred - y_true)^2)
```

### MLP class — backpropagation

```python
dW_list, db_list = mlp.backward(y_true)
```

The algorithm applies the chain rule, walking backwards from the output layer:

```python
# Output layer delta — derivative of MSE
delta = (self.a_cache[-1] - y_true) / n_samples

for i in reversed(range(n_layers)):
    dW_list[i] = delta @ self.a_cache[i].T          # weight gradient
    db_list[i] = np.sum(delta, axis=1, keepdims=True)  # bias gradient
    if i > 0:
        delta = self.weights[i].T @ delta            # propagate error
        delta = delta * self.activation_deriv(self.z_cache[i - 1])  # through activation
```

### MLP class — weight update

```python
mlp.update_weights(dW_list, db_list, learning_rate=0.05)
# w = w - lr * gradient
```

### Batch training loop

```python
from src.trainer import train_batch

loss_history = train_batch(
    mlp,
    x_train=result["x_train"],
    y_train=result["y_train"],
    n_epochs=1000,
    learning_rate=0.05,
)
# loss_history is a list of floats, one MSE per epoch
```

Each epoch runs one full forward → loss → backward → update cycle using **all** training samples at once.

---

## Step 3 — Evaluation

**File:** `src/evaluator.py`

### Predicting in original scale

The network was trained on normalised data, so predictions must be denormalised before comparing to the original y values.

```python
from src.evaluator import predict

y_pred_original = predict(
    mlp,
    x_raw,            # raw (unnormalised) x values
    result["x_mean"], result["x_std"],
    result["y_mean"], result["y_std"],
)
```

Internally this normalises `x_raw`, calls `mlp.forward()`, then denormalises the output.

### Metrics

```python
from src.evaluator import compute_mse, compute_r2

mse = compute_mse(y_pred, y_true)
# MSE = mean((y_pred - y_true)^2)

r2 = compute_r2(y_pred, y_true)
# R² = 1 - SS_residual / SS_total
# 1.0 = perfect, 0.0 = predicts the mean, <0 = worse than mean
```

### Fit diagnosis

```python
from src.evaluator import assess_fit

diagnosis, message = assess_fit(train_mse, test_mse)
# diagnosis is one of: 'underfitting', 'overfitting', 'good fit'
```

| Diagnosis | Condition |
|---|---|
| Underfitting | `train_mse > 0.5` — model hasn't learned the data |
| Overfitting | `test_mse > 2.5 × train_mse` — memorised training set |
| Good fit | Both MSEs are low and close together |

### Visualisation

```python
from src.evaluator import plot_fit, plot_loss_curve

# Fit plot — scatter of train/test + smooth model curve
fig = plot_fit(x_train, y_train, x_test, y_test, x_curve, y_curve)
fig.savefig("fit.png")

# Training loss over epochs (log scale)
fig = plot_loss_curve(loss_history, title="Training loss")
fig.savefig("loss.png")
```

---

## Step 4 — Online (stochastic) training

**File:** `src/trainer.py`

Online training updates the weights after **every single sample**, whereas batch training uses all samples at once.

```python
from src.trainer import train_online

loss_history = train_online(
    mlp,
    x_train=result["x_train"],
    y_train=result["y_train"],
    n_epochs=500,
    learning_rate=0.005,   # smaller lr needed — gradients are noisier
    seed=42,
)
```

Inside each epoch, the samples are visited in a random order:

```python
def train_online_one_epoch(mlp, x, y, learning_rate, rng):
    n_samples = x.shape[1]
    shuffled_indices = rng.permutation(n_samples)   # random order each epoch
    total_loss = 0.0
    for i in shuffled_indices:
        x_i = x[:, i:i + 1]   # single sample, shape (1, 1)
        y_i = y[:, i:i + 1]
        y_pred = mlp.forward(x_i)
        total_loss += mlp.compute_loss(y_pred, y_i)
        dW_list, db_list = mlp.backward(y_i)
        mlp.update_weights(dW_list, db_list, learning_rate)
    return total_loss / n_samples
```

### Batch vs online — key differences

| | Batch GD | Online SGD |
|---|---|---|
| Updates per epoch | 1 | n\_samples |
| Gradient quality | Exact (full data) | Noisy (one point) |
| Convergence | Smooth, stable | Fast early, then oscillates |
| Typical learning rate | 0.01 – 0.1 | 0.001 – 0.01 |

The Flask **Step 4** page trains both methods from identical starting weights and overlays their loss curves so the differences are easy to see.

---

## Step 5 — ReLU activation

**File:** `src/mlp.py`

Switching to ReLU only requires changing the `activation` argument:

```python
mlp = MLP(layer_sizes=[1, 32, 32, 1], activation="relu", seed=42)
```

ReLU networks produce **piecewise-linear** approximations — the model curve is made of straight line segments. More hidden neurons means more segments and a smoother fit.

```
ReLU(z)  = max(0, z)
ReLU′(z) = 1  if z > 0
           0  otherwise
```

Unlike tanh/sigmoid, ReLU does not squash large values, which avoids the vanishing gradient problem in deep networks. However, neurons whose pre-activation is always negative become permanently inactive ("dying ReLU") — this is why a smaller learning rate is recommended for ReLU.

---

## CI/CD pipeline

**File:** `.github/workflows/ci.yml`

Every push and pull request to `main` triggers the pipeline automatically.

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: flake8 src/ tests/ --max-line-length=100
      - run: pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml
      - uses: actions/upload-artifact@v4
        with:
          name: coverage-html
          path: htmlcov/
```

**What each step does:**

| Step | Purpose |
|---|---|
| `checkout` | Fetches the repository onto the runner |
| `setup-python` | Pins Python 3.11 so the environment is reproducible |
| `pip install` | Installs exact dependency versions from `requirements.txt` |
| `flake8` | Catches style errors and unused imports before a human reviewer sees them |
| `pytest` | Runs all 93 unit tests and measures line coverage |
| `upload-artifact` | Saves the HTML coverage report as a downloadable artifact on the Actions page |

If any test fails or flake8 reports an error, GitHub marks the push with a ❌ and blocks merging.

**Reading the coverage report** — open `htmlcov/index.html` from the artifact. Each source file is colour-coded: green lines were executed by at least one test, red lines were not. The current suite achieves **100 % line coverage** across all four source modules.

---

## Test suite overview

```
tests/
├── conftest.py            # matplotlib.use("Agg") — no display needed in CI
├── test_data_loader.py    # 25 tests
├── test_mlp.py            # 47 tests
└── test_evaluator.py      # 21 tests
```

Each test class targets one function so a failure points directly at the broken piece.

```
TestReadFile              — FileNotFoundError, line count, string type
TestParseLines            — blank line skipping, float conversion, equal lengths
TestComputeMeanStd        — correct mean, correct std, std never zero
TestNormalize             — mean → 0, std → 1, shape preserved
TestDenormalize           — round-trip restores original values
TestSplitTrainTest        — ratio, reproducibility, no overlap between sets
TestLoadData              — all keys present, sizes add up, stats are scalars

TestInitializeWeights     — shapes, bias zeros, reproducibility
TestMLPForward            — output shape, cache lengths, finite values
TestComputeLoss           — scalar, zero for perfect, correct MSE formula
TestMLPBackward           — gradient shapes match weights, finite gradients
TestUpdateWeights         — weights change, zero lr leaves weights unchanged
TestTrainBatch            — history length, loss decreases, non-negative values
TestTrainOnline           — same checks plus per-sample update and seed reproducibility
TestActivations           — tanh/sigmoid/ReLU known values, unknown name raises ValueError
TestPrepareInputs         — 1-D → (1, n) shape, values preserved

TestComputeMse            — correct value, symmetric, non-negative, float return
TestComputeR2             — 1.0 for perfect, 0.0 for mean predictor, negative for terrible
TestPredict               — length matches input, 1-D output, finite values
TestAssessFit             — all three diagnoses, message is non-empty string
TestPlots                 — Figure returned, no crash on single-epoch input
```

Run the full suite in one command:

```bash
pytest tests/ -v --tb=short
```
