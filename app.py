"""
app.py

Flask web application that showcases all 5 steps of the MLP project.
Each route corresponds to one step in the task description.

The Agg backend is set before any other matplotlib import so that plots
render correctly in a headless environment (no display attached).
"""

import matplotlib
matplotlib.use("Agg")  # must come before any other matplotlib import

import io                                                                    # noqa: E402
import os                                                                    # noqa: E402
import base64                                                                # noqa: E402

import numpy as np                                                           # noqa: E402
import matplotlib.pyplot as plt                                              # noqa: E402
from flask import Flask, render_template, request, redirect, url_for        # noqa: E402

from src.data_loader import load_data, denormalize                          # noqa: E402
from src.mlp import MLP                                                      # noqa: E402
from src.trainer import train_batch, train_online                            # noqa: E402
from src.evaluator import (compute_mse, compute_r2, predict,               # noqa: E402
                           assess_fit, plot_fit, plot_loss_curve)


app = Flask(__name__)

# Global training state — shared across requests so the evaluate page can
# reuse the model that was trained on the train page without re-training.
state = {
    "mlp": None,
    "data": None,
    "loss_history": None,
    "config": None,
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASETS = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".txt"))


# Helpers


def generate_tips(diagnosis, train_r2, test_r2, config):
    """
    Return a list of actionable tip dicts based on the current model's performance.

    Each tip has:
        title   -- short label shown in bold
        detail  -- one sentence explaining why this helps
        action  -- what the user should change on the Train page

    We look at the diagnosis first, then check individual metric values to add
    extra context-specific advice on top.
    """
    tips = []

    if diagnosis == "underfitting":
        tips.append({
            "title": "Add more hidden neurons",
            "detail": "The network is too small to capture the pattern in the data.",
            "action": "Increase 'Hidden neurons per layer' on the Train page (try 32 or 64).",
        })
        tips.append({
            "title": "Add a second hidden layer",
            "detail": "Deeper networks can represent more complex functions.",
            "action": "Set 'Number of hidden layers' to 2.",
        })
        tips.append({
            "title": "Train for more epochs",
            "detail": "The model may not have had enough iterations to converge yet.",
            "action": "Double the epoch count and retrain.",
        })
        if config and config.get("lr", 0) < 0.01:
            tips.append({
                "title": "Try a higher learning rate",
                "detail": "A very small learning rate slows convergence significantly.",
                "action": f"Current lr is {config['lr']} — try 0.05 or 0.1.",
            })

    elif diagnosis == "overfitting":
        tips.append({
            "title": "Reduce network capacity",
            "detail": "A large network memorises the training data instead of the pattern.",
            "action": "Decrease 'Hidden neurons per layer' or use only 1 hidden layer.",
        })
        fewer = max(50, config["epochs"] // 3)
        tips.append({
            "title": "Train for fewer epochs",
            "detail": "Training too long lets the network overfit. Stop earlier.",
            "action": f"Try {fewer} epochs instead of {config['epochs']}.",
        })
        tips.append({
            "title": "Lower the learning rate",
            "detail": "A smaller lr produces smoother updates and often generalises better.",
            "action": "Try a learning rate 5× smaller than the current one.",
        })

    else:  # good fit
        tips.append({
            "title": "Try a different dataset",
            "detail": "Some datasets are harder to approximate — test the model's limits.",
            "action": "Pick a different file from the dataset dropdown and retrain.",
        })
        if train_r2 < 0.95:
            tips.append({
                "title": "There is still room to improve R²",
                "detail": (
                    f"Train R² is {train_r2:.3f}. More neurons or epochs may push it closer to 1."
                ),
                "action": "Add 8–16 more hidden neurons and increase epochs by 50%.",
            })
        tips.append({
            "title": "Compare with online training",
            "detail": "Online SGD often converges faster — see if it matches this fit.",
            "action": "Go to Step 4 — Batch vs Online and run the comparison.",
        })
        tips.append({
            "title": "Try ReLU",
            "detail": "ReLU gives a piecewise-linear curve — useful to compare with smooth ones.",
            "action": "Go to Step 5 — ReLU Network and train on the same dataset.",
        })

    # Universal tip when test generalisation is weak
    if test_r2 < 0.5 and diagnosis != "overfitting":
        tips.append({
            "title": "Test R² is low — generalisation is poor",
            "detail": (
                f"Test R² of {test_r2:.3f} means less than half the variance is explained "
                "on unseen data."
            ),
            "action": "Check the fit plot — if the curve misses test points, add more capacity.",
        })

    return tips


def fig_to_base64(fig):
    """Save a matplotlib Figure to a base64 PNG string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def raw_from_data(data_result):
    """
    Denormalize all four arrays in a load_data() result dict back to original scale.

    Returns x_train_raw, x_test_raw, y_train_raw, y_test_raw as 1-D arrays.
    """
    x_train = denormalize(data_result["x_train"], data_result["x_mean"], data_result["x_std"])
    x_test = denormalize(data_result["x_test"], data_result["x_mean"], data_result["x_std"])
    y_train = denormalize(data_result["y_train"], data_result["y_mean"], data_result["y_std"])
    y_test = denormalize(data_result["y_test"], data_result["y_mean"], data_result["y_std"])
    return x_train, x_test, y_train, y_test


def make_curve(mlp, data_result, n_points=250):
    """
    Generate a smooth (x, y) curve by running the trained model over the full x range.

    We use the raw (denormalized) x range so the curve covers all the data.
    """
    x_train_r, x_test_r, _, _ = raw_from_data(data_result)
    x_all = np.concatenate([x_train_r, x_test_r])
    x_curve = np.linspace(x_all.min(), x_all.max(), n_points)
    y_curve = predict(mlp, x_curve,
                      data_result["x_mean"], data_result["x_std"],
                      data_result["y_mean"], data_result["y_std"])
    return x_curve, y_curve


def scatter_plot(x_train, y_train, x_test, y_test, title):
    """
    Create a scatter plot of train and test points and return it as base64 PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_train, y_train, s=22, alpha=0.8, color="#3b82f6",
               label=f"Train  ({len(x_train)} pts)")
    ax.scatter(x_test, y_test, s=22, alpha=0.8, color="#ef4444",
               label=f"Test  ({len(x_test)} pts)")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig_to_base64(fig)


# Routes


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data", methods=["GET", "POST"])
def data():
    dataset = request.form.get("dataset", "dane1.txt")
    filepath = os.path.join(DATA_DIR, dataset)

    result = load_data(filepath)
    x_tr, x_te, y_tr, y_te = raw_from_data(result)

    plot_img = scatter_plot(x_tr, y_tr, x_te, y_te, f"Dataset: {dataset}")

    stats = {
        "n_total":  len(x_tr) + len(x_te),
        "n_train":  len(x_tr),
        "n_test":   len(x_te),
        "x_range":  f"[{min(x_tr.min(), x_te.min()):.3f},  {max(x_tr.max(), x_te.max()):.3f}]",
        "y_range":  f"[{min(y_tr.min(), y_te.min()):.3f},  {max(y_tr.max(), y_te.max()):.3f}]",
        "x_mean":   f"{result['x_mean']:.4f}",
        "x_std":    f"{result['x_std']:.4f}",
        "y_mean":   f"{result['y_mean']:.4f}",
        "y_std":    f"{result['y_std']:.4f}",
    }

    return render_template("data.html", datasets=DATASETS, selected=dataset,
                           plot_img=plot_img, stats=stats)


@app.route("/train", methods=["GET", "POST"])
def train():
    plot_img = None
    config = None

    defaults = dict(dataset="dane1.txt", activation="tanh",
                    hidden_size=16, n_hidden=1, epochs=1000, lr=0.05)

    if request.method == "POST":
        dataset = request.form.get("dataset", defaults["dataset"])
        activation = request.form.get("activation", defaults["activation"])
        hidden_size = int(request.form.get("hidden_size", defaults["hidden_size"]))
        n_hidden = int(request.form.get("n_hidden", defaults["n_hidden"]))
        epochs = int(request.form.get("epochs", defaults["epochs"]))
        lr = float(request.form.get("lr", defaults["lr"]))

        filepath = os.path.join(DATA_DIR, dataset)
        data_result = load_data(filepath)

        layer_sizes = [1] + [hidden_size] * n_hidden + [1]

        mlp = MLP(layer_sizes=layer_sizes, activation=activation, seed=42)
        history = train_batch(mlp, data_result["x_train"], data_result["y_train"],
                              n_epochs=epochs, learning_rate=lr)

        # Persist for the evaluate page
        state["mlp"] = mlp
        state["data"] = data_result
        state["loss_history"] = history
        state["config"] = dict(dataset=dataset, activation=activation,
                               layer_sizes=layer_sizes, epochs=epochs, lr=lr)

        fig = plot_loss_curve(history, title=f"Batch training — {activation}  {layer_sizes}")
        plot_img = fig_to_base64(fig)

        config = dict(**state["config"],
                      final_loss=f"{history[-1]:.6f}",
                      arch=" → ".join(str(s) for s in layer_sizes))

        defaults = dict(dataset=dataset, activation=activation,
                        hidden_size=hidden_size, n_hidden=n_hidden, epochs=epochs, lr=lr)

    return render_template("train.html", datasets=DATASETS, defaults=defaults,
                           plot_img=plot_img, config=config)


@app.route("/evaluate")
def evaluate():
    if state["mlp"] is None:
        return render_template("evaluate.html", no_model=True)

    mlp = state["mlp"]
    data_result = state["data"]

    x_tr, x_te, y_tr, y_te = raw_from_data(data_result)

    # Predictions in original scale
    y_tr_pred = predict(mlp, x_tr, data_result["x_mean"], data_result["x_std"],
                        data_result["y_mean"], data_result["y_std"])
    y_te_pred = predict(mlp, x_te, data_result["x_mean"], data_result["x_std"],
                        data_result["y_mean"], data_result["y_std"])

    # Smooth model curve for the fit plot
    x_curve, y_curve = make_curve(mlp, data_result)

    train_mse = compute_mse(y_tr_pred, y_tr)
    test_mse = compute_mse(y_te_pred, y_te)
    train_r2 = compute_r2(y_tr_pred, y_tr)
    test_r2 = compute_r2(y_te_pred, y_te)
    diagnosis, message = assess_fit(train_mse, test_mse)

    fig = plot_fit(x_tr, y_tr, x_te, y_te, x_curve, y_curve,
                   title="Model fit — original scale")
    fit_img = fig_to_base64(fig)

    metrics = dict(
        train_mse=f"{train_mse:.4f}", test_mse=f"{test_mse:.4f}",
        train_r2=f"{train_r2:.4f}", test_r2=f"{test_r2:.4f}",
        diagnosis=diagnosis, message=message,
    )

    tips = generate_tips(diagnosis, train_r2, test_r2, state["config"])

    return render_template("evaluate.html", no_model=False,
                           fit_img=fit_img, metrics=metrics,
                           config=state["config"], tips=tips)


@app.route("/compare", methods=["GET", "POST"])
def compare():
    batch_img = online_img = overlay_img = None
    comparison = None

    defaults = dict(dataset="dane1.txt", hidden_size=16,
                    epochs=500, batch_lr=0.05, online_lr=0.005)

    if request.method == "POST":
        dataset = request.form.get("dataset", defaults["dataset"])
        hidden_size = int(request.form.get("hidden_size", defaults["hidden_size"]))
        epochs = int(request.form.get("epochs", defaults["epochs"]))
        batch_lr = float(request.form.get("batch_lr", defaults["batch_lr"]))
        online_lr = float(request.form.get("online_lr", defaults["online_lr"]))

        filepath = os.path.join(DATA_DIR, dataset)
        data_result = load_data(filepath)
        layer_sizes = [1, hidden_size, 1]

        # Both models start from the exact same random weights (seed=42)
        mlp_batch = MLP(layer_sizes=layer_sizes, activation="tanh", seed=42)
        history_batch = train_batch(mlp_batch, data_result["x_train"], data_result["y_train"],
                                    n_epochs=epochs, learning_rate=batch_lr)

        mlp_online = MLP(layer_sizes=layer_sizes, activation="tanh", seed=42)
        history_online = train_online(mlp_online, data_result["x_train"], data_result["y_train"],
                                      n_epochs=epochs, learning_rate=online_lr, seed=42)

        fig_b = plot_loss_curve(history_batch, title=f"Batch GD  (lr={batch_lr})")
        batch_img = fig_to_base64(fig_b)

        fig_o = plot_loss_curve(history_online, title=f"Online SGD  (lr={online_lr})")
        online_img = fig_to_base64(fig_o)

        # Overlay comparison chart
        fig, ax = plt.subplots(figsize=(9, 5))
        ep = range(1, epochs + 1)
        ax.plot(ep, history_batch, label=f"Batch GD  (lr={batch_lr})",
                color="#3b82f6", linewidth=2)
        ax.plot(ep, history_online, label=f"Online SGD  (lr={online_lr})",
                color="#f59e0b", linewidth=2, alpha=0.9)
        ax.set_yscale("log")
        ax.set_title("Batch vs Online — training loss (log scale)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        overlay_img = fig_to_base64(fig)

        n_samples = len(data_result["x_train"])
        comparison = dict(
            batch_final=f"{history_batch[-1]:.6f}",
            online_final=f"{history_online[-1]:.6f}",
            batch_updates=f"{epochs:,}",
            online_updates=f"{epochs * n_samples:,}",
        )

        defaults = dict(dataset=dataset, hidden_size=hidden_size,
                        epochs=epochs, batch_lr=batch_lr, online_lr=online_lr)

    return render_template("compare.html", datasets=DATASETS, defaults=defaults,
                           batch_img=batch_img, online_img=online_img,
                           overlay_img=overlay_img, comparison=comparison)


@app.route("/relu", methods=["GET", "POST"])
def relu():
    loss_img = fit_img = None
    config = None

    defaults = dict(dataset="dane1.txt", hidden_size=32, epochs=1000, lr=0.01)

    if request.method == "POST":
        dataset = request.form.get("dataset", defaults["dataset"])
        hidden_size = int(request.form.get("hidden_size", defaults["hidden_size"]))
        epochs = int(request.form.get("epochs", defaults["epochs"]))
        lr = float(request.form.get("lr", defaults["lr"]))

        filepath = os.path.join(DATA_DIR, dataset)
        data_result = load_data(filepath)

        # Two hidden layers — ReLU networks often need more neurons to approximate curves
        layer_sizes = [1, hidden_size, hidden_size, 1]
        mlp = MLP(layer_sizes=layer_sizes, activation="relu", seed=42)
        history = train_batch(mlp, data_result["x_train"], data_result["y_train"],
                              n_epochs=epochs, learning_rate=lr)

        fig = plot_loss_curve(history, title="ReLU network — training loss")
        loss_img = fig_to_base64(fig)

        x_tr, x_te, y_tr, y_te = raw_from_data(data_result)
        x_curve, y_curve = make_curve(mlp, data_result)

        fig2 = plot_fit(x_tr, y_tr, x_te, y_te, x_curve, y_curve,
                        title="ReLU network fit")
        fit_img = fig_to_base64(fig2)

        config = dict(
            arch=" → ".join(str(s) for s in layer_sizes),
            epochs=epochs, lr=lr,
            final_loss=f"{history[-1]:.6f}",
        )

        defaults = dict(dataset=dataset, hidden_size=hidden_size, epochs=epochs, lr=lr)

    return render_template("relu.html", datasets=DATASETS, defaults=defaults,
                           loss_img=loss_img, fit_img=fit_img, config=config)


@app.route("/reset")
def reset():
    """
    Clear all training state so the app is back to a clean slate.

    Useful when presenting — one click wipes the trained model, data, and
    loss history so the demo can start fresh without restarting the server.
    """
    state["mlp"] = None
    state["data"] = None
    state["loss_history"] = None
    state["config"] = None
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
