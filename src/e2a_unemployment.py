"""Experiment 2a: does the black box win at forecasting unemployment?

Task: forecast the change in the US unemployment rate over the next 12 months,
as in the Bank of England study of Buckmann and Joseph (2023), using the
FRED-MD panel of monthly macroeconomic series.

Four forecasters are compared on the same months:
  - naive: unemployment does not change,
  - autoregressive model (interpretable): uses the last six values of the
    unemployment rate only,
  - sparse linear model (interpretable): may pick at most 10 series of the panel,
  - gradient boosting (black box): uses every series of the panel.

Evaluation uses a rolling origin: every forecast is made by a model fitted only
on data that were available 12 months before the month being forecast.

Run:  python src/e2a_unemployment.py
Writes outputs/e2a_results.csv, e2a_gaps.csv, e2a_forecasts.csv, e2a_models.txt
and e2a_figure.png.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.linear_model import lasso_path

from common import (COLOR_BLACK_BOX, COLOR_GUIDE, COLOR_HIGHLIGHT, COLOR_INTERPRETABLE,
                    COLOR_INTERPRETABLE_LIGHT, COLOR_REFERENCE, LIGHTGBM_REPRODUCIBLE, N_BOOTSTRAP,
                    OUTPUT_DIR, SEED, block_bootstrap_indices, percentile_interval)
from data import load_fred_md

HORIZON = 12                 # months ahead
AR_LAGS = 6                  # the autoregressive model sees the last six unemployment rates
SMOOTHING = 3                # each panel series is averaged over its last three months
FIRST_TEST_MONTH = "1990-01"
REFIT_EVERY = 12             # models are re-estimated once a year
VALIDATION_MONTHS = 120      # last ten years of the training window, used to choose settings
COVID_FIRST, COVID_LAST = "2020-03", "2021-12"
BLOCK_LENGTH = 24            # months per block in the block bootstrap

NAIVE, AR, SPARSE, GBM = "Naive (no change)", "Autoregressive model", "Sparse linear model", "Gradient boosting"
METHODS = [NAIVE, AR, SPARSE, GBM]
# Pairs of methods compared in e2a_gaps.csv, in the order of its rows.
COMPARISONS = [(GBM, SPARSE), (GBM, AR), (GBM, NAIVE), (SPARSE, AR), (SPARSE, NAIVE), (AR, NAIVE)]

# Six candidate settings for each tuned model, so that both get the same tuning effort.
# The sparse model may use at most 10 series, our criterion for an interpretable linear model.
SPARSE_SIZES = [1, 2, 3, 5, 8, 10]      # number of series the sparse model may use
GBM_GRID = [dict(num_leaves=leaves, n_estimators=trees) for leaves in (2, 4, 8) for trees in (100, 300)]


# ----------------------------------------------------------------------------------
# The three trained models. Each `fit_*` function receives training inputs X and
# targets (12-month changes in unemployment). `fit_ar` returns the coefficients,
# constant first; `fit_sparse` returns ((constant, coefficients), number of series
# allowed); `fit_gbm` returns (fitted model, chosen setting).
# ----------------------------------------------------------------------------------

def fit_ar(X, target):
    """Least-squares regression of the 12-month change on the last six unemployment rates."""
    design = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coef


def lasso_with_at_most(X, target, sizes):
    """For each size k, the Lasso solution that uses at most k variables.

    The Lasso is a linear regression with a penalty that sets most coefficients
    exactly to zero. The stronger the penalty, the fewer variables remain. We
    scan penalties from strong to weak and keep, for each k, the weakest
    penalty that still uses at most k variables. Inputs are standardised first
    so that the penalty treats all series alike.
    """
    mean, std = X.mean(axis=0), X.std(axis=0)
    Z = (X - mean) / std
    _, coefs, _ = lasso_path(Z, target - target.mean(), n_alphas=100, eps=1e-3)
    n_used = (coefs != 0).sum(axis=0)
    solutions = {}
    for k in sizes:
        column = np.flatnonzero(n_used <= k)[-1]
        beta = coefs[:, column] / std                      # back to the original units
        solutions[k] = (target.mean() - beta @ mean, beta)  # (intercept, coefficients)
    return solutions


def fit_sparse(X, target):
    """Sparse linear model; the number of series (1 to 10) is chosen on the last ten training years."""
    cut = len(target) - VALIDATION_MONTHS
    candidates = lasso_with_at_most(X[:cut], target[:cut], SPARSE_SIZES)
    errors = {k: np.mean((b0 + X[cut:] @ b - target[cut:]) ** 2) for k, (b0, b) in candidates.items()}
    best_size = min(errors, key=errors.get)
    return lasso_with_at_most(X, target, [best_size])[best_size], best_size


def new_gbm(settings):
    """Gradient boosting with the fixed settings of E2a and one candidate setting of GBM_GRID."""
    return LGBMRegressor(learning_rate=0.05, min_child_samples=10, random_state=SEED, **LIGHTGBM_REPRODUCIBLE,
                         **settings)


def fit_gbm(X, target):
    """Gradient boosting; the setting is chosen on the last ten training years."""
    cut = len(target) - VALIDATION_MONTHS
    errors = [np.mean((new_gbm(s).fit(X[:cut], target[:cut]).predict(X[cut:]) - target[cut:]) ** 2)
              for s in GBM_GRID]
    settings = GBM_GRID[int(np.argmin(errors))]
    return new_gbm(settings).fit(X, target), settings


# ----------------------------------------------------------------------------------

def rmse(errors):
    """Root mean squared error of an array of forecast errors."""
    return float(np.sqrt(np.mean(errors ** 2)))


def prepare_inputs(unrate, panel):
    """Inputs and target of every month, as known at the end of that month.

    The sparse model and the gradient boosting see the recent trend of every
    series of the panel (average of its last three monthly values) and the
    current unemployment rate; the autoregressive model sees the last six
    unemployment rates. Row i describes month i, and its target is the change
    in unemployment over the following 12 months (unknown for the last 12 rows).
    Returns the months, the unemployment rate, the two input arrays, the names
    of the panel inputs and the target.
    """
    features = panel.rolling(SMOOTHING).mean()
    features.insert(0, "UNRATE_level", unrate)
    features = features.iloc[AR_LAGS - 1:]
    months = features.index
    y = unrate.loc[months].to_numpy()
    X_panel = features.to_numpy()
    full = unrate.to_numpy()
    offset = len(full) - len(y)
    # Column k holds the unemployment rate k months before the month of the row.
    X_ar = np.column_stack([full[offset - k: len(full) - k] for k in range(AR_LAGS)])  # most recent first
    names = list(features.columns)

    change = np.full(len(y), np.nan)
    change[:-HORIZON] = y[HORIZON:] - y[:-HORIZON]
    return months, y, X_panel, X_ar, names, change


def rolling_forecasts(months, y, X_panel, X_ar, change, first_target):
    """Forecasts for every month from FIRST_TEST_MONTH on, each made 12 months earlier.

    The models are refitted once a year, at the first forecast and then every
    REFIT_EVERY months, and kept in between. Returns one row per forecast month,
    with the actual rate and the forecast of every method, and the settings
    chosen at each refit.
    """
    rows, chosen = [], []
    for target_month in range(first_target, len(y)):
        origin = target_month - HORIZON          # the forecast is made at the end of this month
        if (target_month - first_target) % REFIT_EVERY == 0:
            # Only rows whose 12-month outcome is already known at the origin can be used.
            train = slice(0, origin - HORIZON + 1)
            ar_coef = fit_ar(X_ar[train], change[train])
            (sparse_b0, sparse_b), size = fit_sparse(X_panel[train], change[train])
            gbm, settings = fit_gbm(X_panel[train], change[train])
            chosen.append((size, str(settings)))
        predicted_change = {
            NAIVE: 0.0,
            AR: ar_coef[0] + ar_coef[1:] @ X_ar[origin],
            SPARSE: sparse_b0 + X_panel[origin] @ sparse_b,
            GBM: gbm.predict(X_panel[origin].reshape(1, -1))[0],
        }
        row = {"month": str(months[target_month]), "actual": y[target_month]}
        row.update({m: y[origin] + predicted_change[m] for m in METHODS})
        rows.append(row)
    return pd.DataFrame(rows), chosen


def errors_with_intervals(forecasts):
    """RMSE of every method and RMSE gaps, over all months and without the Covid period.

    Forecast errors of neighbouring months are related, because 12-month windows
    overlap, so the months are resampled in blocks of BLOCK_LENGTH. One generator
    serves both periods, in the order of `periods`: the draws for the period
    without Covid follow those for all months. Every method is scored on the
    same resamples, so that the gaps are paired.
    """
    covid = forecasts["month"].between(COVID_FIRST, COVID_LAST).to_numpy()
    periods = {"all months": np.ones(len(forecasts), bool), "without Covid period": ~covid}
    errors = {m: (forecasts[m] - forecasts["actual"]).to_numpy() for m in METHODS}

    rng = np.random.default_rng(SEED)
    result_rows, gap_rows = [], []
    for period, mask in periods.items():
        e = {m: errors[m][mask] for m in METHODS}
        n = int(mask.sum())
        boot = {m: np.empty(N_BOOTSTRAP) for m in METHODS}
        for b in range(N_BOOTSTRAP):
            idx = block_bootstrap_indices(n, BLOCK_LENGTH, rng)
            for m in METHODS:
                boot[m][b] = rmse(e[m][idx])
        for m in METHODS:
            low, high = percentile_interval(boot[m])
            result_rows.append({"period": period, "months": n, "method": m, "rmse": rmse(e[m]),
                                "rmse_low": low, "rmse_high": high, "mae": float(np.mean(np.abs(e[m])))})
        for first, second in COMPARISONS:
            low, high = percentile_interval(boot[first] - boot[second])
            gap_rows.append({"period": period, "comparison": f"{first} minus {second}",
                             "rmse_gap": rmse(e[first]) - rmse(e[second]),
                             "rmse_gap_low": low, "rmse_gap_high": high})
    return pd.DataFrame(result_rows), pd.DataFrame(gap_rows)


def describe_final_models(X_ar, X_panel, change, names, chosen):
    """The three models fitted on all the data, in readable form: the lines of e2a_models.txt."""
    known = slice(0, len(change) - HORIZON)
    final_ar = fit_ar(X_ar[known], change[known])
    (b0, b), _ = fit_sparse(X_panel[known], change[known])
    final_gbm, final_settings = fit_gbm(X_panel[known], change[known])
    scale = X_panel[known].std(axis=0)
    lines = ["== Autoregressive model ==",
             f"change over the next 12 months = {final_ar[0]:+.3f}"]
    lines += [f"    {c:+.3f} x unemployment rate {k} month(s) ago" if k else f"    {c:+.3f} x current unemployment rate"
              for k, c in enumerate(final_ar[1:])]
    lines += ["", f"== Sparse linear model ({int((b != 0).sum())} series used) ==",
              f"change over the next 12 months = {b0:+.3f}",
              "    (effect of a one-standard-deviation rise of each series, largest first)"]
    used = sorted(np.flatnonzero(b), key=lambda j: -abs(b[j] * scale[j]))
    lines += [f"    {b[j] * scale[j]:+.3f} for {names[j]}" for j in used]
    sizes = pd.Series([c[0] for c in chosen]).value_counts().sort_index()
    lines += ["number of series chosen over the yearly refits: "
              + ", ".join(f"{k} series: {n} times" for k, n in sizes.items())]
    importance = np.abs(shap.TreeExplainer(final_gbm).shap_values(X_panel[known])).mean(axis=0)
    top = np.argsort(-importance)[:10]
    n_used = int((final_gbm.feature_importances_ > 0).sum())
    lines += ["", "== Gradient boosting ==", f"chosen settings: {final_settings}",
              f"sum of {final_settings['n_estimators']} trees using {n_used} of the {len(names)} inputs: "
              "not readable directly",
              "ten inputs with the largest average absolute SHAP value:"]
    lines += [f"    {importance[j]:.3f} for {names[j]}" for j in top]
    lines += ["settings chosen over the yearly refits: "
              + ", ".join(f"{k}: {n} times" for k, n in pd.Series([c[1] for c in chosen]).value_counts().items())]
    return lines


def main():
    """Forecast every month since 1990, measure the errors, describe the models, then save, print and draw."""
    warnings.filterwarnings("ignore")   # hide library warnings, so that the printed tables stay readable
    OUTPUT_DIR.mkdir(exist_ok=True)

    unrate, panel = load_fred_md()
    months, y, X_panel, X_ar, names, change = prepare_inputs(unrate, panel)
    first_target = months.get_loc(pd.Period(FIRST_TEST_MONTH, "M"))
    print(f"{X_panel.shape[1]} inputs, months {months[0]} to {months[-1]}; "
          f"forecasts for {FIRST_TEST_MONTH} to {months[-1]} ({len(y) - first_target} months)")

    # ---- 1. Rolling-origin forecasts ---------------------------------------------------
    forecasts, chosen = rolling_forecasts(months, y, X_panel, X_ar, change, first_target)
    forecasts.round(4).to_csv(OUTPUT_DIR / "e2a_forecasts.csv", index=False)

    # ---- 2. Errors, with and without the Covid period ------------------------------------
    results, gaps = errors_with_intervals(forecasts)
    results.round(4).to_csv(OUTPUT_DIR / "e2a_results.csv", index=False)
    gaps.round(4).to_csv(OUTPUT_DIR / "e2a_gaps.csv", index=False)

    # ---- 3. The models in readable form (fitted on all the data) ----------------------------
    lines = describe_final_models(X_ar, X_panel, change, names, chosen)
    (OUTPUT_DIR / "e2a_models.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\nForecast errors in percentage points of unemployment (95% block-bootstrap intervals)")
    print(results.round(3).to_string(index=False))
    print("\nGaps in RMSE (negative = the first method is better)")
    print(gaps.round(3).to_string(index=False))
    print("\n" + "\n".join(lines))

    make_figure(forecasts, results)
    print(f"\nFiles written in {OUTPUT_DIR}")


def make_figure(forecasts, results):
    """Quick-look figure of E2a: the forecasts against what happened, and the error of each method.

    The report uses figure 2, drawn by report_figures.py, instead.
    """
    colors = {NAIVE: COLOR_REFERENCE, AR: COLOR_INTERPRETABLE_LIGHT, SPARSE: COLOR_INTERPRETABLE, GBM: COLOR_BLACK_BOX}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})

    # Left: forecasts made 12 months earlier, against what happened.
    ax = axes[0]
    dates = pd.PeriodIndex(forecasts["month"], freq="M").to_timestamp()
    ax.plot(dates, forecasts["actual"], color=COLOR_HIGHLIGHT, linewidth=2, label="actual")
    ax.plot(dates, forecasts[SPARSE], color=colors[SPARSE], linewidth=1.2, label=SPARSE)
    ax.plot(dates, forecasts[GBM], "--", color=colors[GBM], linewidth=1.2, label=GBM)
    ax.axvspan(pd.Timestamp(COVID_FIRST), pd.Timestamp(COVID_LAST) + pd.offsets.MonthEnd(0),
               color="#eeeeee", zorder=0)
    ax.set_ylim(2, 16)
    ax.set_ylabel("US unemployment rate (%)")
    ax.set_title("Forecasts made 12 months earlier (grey: Covid period)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # Right: forecast error outside the Covid period.
    ax = axes[1]
    part = results[results["period"] == "without Covid period"].set_index("method").loc[METHODS]
    x = np.arange(len(METHODS))
    ax.bar(x, part["rmse"], color=[colors[m] for m in METHODS],
           yerr=[part["rmse"] - part["rmse_low"], part["rmse_high"] - part["rmse"]],
           error_kw={"capsize": 3, "ecolor": COLOR_GUIDE})
    for xi, value in zip(x, part["rmse"]):
        ax.text(xi, 0.03, f"{value:.2f}", ha="center", va="bottom", color="white", fontsize=9)
    ax.set_xticks(x, ["Naive", "Auto-\nregressive", "Sparse\nlinear", "Gradient\nboosting"])
    ax.set_ylabel("RMSE (percentage points)")
    ax.set_title("Error since 1990, without the Covid period", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "e2a_figure.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
