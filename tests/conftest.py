import matplotlib

# Switch to the non-interactive Agg backend before any test imports matplotlib.
# This prevents crashes in CI where there is no display attached.
matplotlib.use("Agg")
