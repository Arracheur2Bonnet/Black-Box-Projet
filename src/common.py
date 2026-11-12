"""Shared settings and helpers used by all experiments."""

from pathlib import Path

import numpy as np

# One seed for every random step, so that two runs give the same numbers.
SEED = 2026

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"        # downloaded files, ignored by git
OUTPUT_DIR = ROOT / "outputs"   # figures and result files, ignored by git

# Colours used in every figure.
COLOR_INTERPRETABLE = "#2a7f62"
COLOR_BLACK_BOX = "#333333"
COLOR_REFERENCE = "#b0b0b0"


def bootstrap_indices(n, n_resamples, seed=SEED):
    """Return an array of shape (n_resamples, n) of row indices drawn with replacement.

    Each row is one bootstrap resample of the n observations. Using the same
    indices for every model keeps the comparison paired: all models are scored
    on exactly the same resampled people.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n))


def percentile_interval(values, level=0.95):
    """Percentile bootstrap interval: the central `level` share of the values."""
    alpha = (1 - level) / 2
    low, high = np.quantile(values, [alpha, 1 - alpha])
    return float(low), float(high)
