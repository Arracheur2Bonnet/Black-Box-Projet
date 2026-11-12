"""Experiment 2b: does the black box win at forecasting electricity demand?

This problem was chosen because a black box is expected to do well on it:
electricity demand depends on temperature in a non-linear way (heating when it
is cold, air conditioning when it is hot), this effect changes with the time of
day and the type of day, and there are tens of thousands of observations.

Task: once a day, forecast the demand of each of the next 48 half-hours in the
state of Victoria (Australia). Four forecasters receive the same information:
  - naive: the demand of the same half-hour one week earlier,
  - linear regression with 9 variables (interpretable),
  - additive model: one curve per input, readable curve by curve,
  - gradient boosting, free to combine the inputs (black box).

Run:  python src/e2b_electricity.py
Writes outputs/e2b_results.csv, e2b_gaps.csv, e2b_peaks.csv, e2b_forecasts.csv,
e2b_curves.csv, e2b_models.txt and e2b_figure.png.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from common import (COLOR_BLACK_BOX, COLOR_INTERPRETABLE, COLOR_REFERENCE, OUTPUT_DIR,
                    SEED, percentile_interval)
from data import load_vic_electricity

PER_DAY = 48                    # half-hours in a forecast day
FIRST_USABLE_DAY = 7            # the first week is needed to build "one week earlier"
FIRST_TEST_DAY = 731            # 1 January 2014 (2012 has 366 days, 2013 has 365)
VALIDATION_DAYS = 56            # last eight weeks of 2013, used once to choose the settings
COMFORT_TEMPERATURE = 18.0      # usual threshold for heating and cooling degrees
N_BOOTSTRAP = 2000
BLOCK_DAYS = 7                  # days per block in the block bootstrap
PEAK_SHARE = 0.01               # "peak" half-hours: the 1% with the highest demand

NAIVE, LINEAR, ADDITIVE, GBM = ("Naive (same time last week)", "Linear regression",
                                "Additive model", "Gradient boosting")
METHODS = [NAIVE, LINEAR, ADDITIVE, GBM]

# Information given to the tree-based models (additive model and gradient boosting).
TREE_INPUTS = ["half_hour", "day_of_week", "workday", "workday_yesterday",
               "temperature", "temperature_yesterday",
               "demand_yesterday", "demand_last_week", "demand_yesterday_mean"]
# The linear regression receives the same information in a form a straight line
# can use. Temperature enters through heating and cooling degrees, the textbook
# transformation, because a straight line cannot bend. The time of day and the
# day of the week are carried by the past demand at the same half-hour: one
# coefficient per half-hour would need 47 more variables, far above our limit
# of 10 variables for an interpretable model.
LINEAR_INPUTS = ["workday", "workday_yesterday",
                 "heating_degrees", "cooling_degrees",
                 "heating_degrees_yesterday", "cooling_degrees_yesterday",
                 "demand_yesterday", "demand_last_week", "demand_yesterday_mean"]

# Six candidate settings for each tree-based model. A tree with two leaves looks
# at one input only, so a sum of such trees is a sum of one curve per input: an
# additive model. Larger trees can combine several inputs.
ADDITIVE_TREES = [250, 500, 1000, 2000, 4000, 8000]
GBM_GRID = [(leaves, trees) for leaves in (8, 31, 127) for trees in (300, 1000)]


def build_inputs(data):
    """Add the model inputs to the raw data. Every input is known when the forecast is made.

    A forecast day is a block of 48 consecutive half-hours, starting at the
    first row of the data. The forecast for a day is made at its start: the
    demand of the previous day and of the day one week earlier is then known.
    The temperature of the forecast day is taken as known for all models, as if
    the weather forecast were perfect.
    """
    d = data.copy()
    d["day"] = np.arange(len(d)) // PER_DAY
    d["half_hour"] = d["time"].dt.hour * 2 + d["time"].dt.minute // 30      # 0 = midnight, 47 = 23:30
    d["day_of_week"] = d["time"].dt.day_of_week                             # 0 = Monday
    d["workday"] = ((d["day_of_week"] < 5) & (d["holiday"] == 0)).astype(int)
    d["workday_yesterday"] = d["workday"].shift(PER_DAY)
    d["temperature_yesterday"] = d["temperature"].shift(PER_DAY)
    d["demand_yesterday"] = d["demand"].shift(PER_DAY)
    d["demand_last_week"] = d["demand"].shift(7 * PER_DAY)
    d["demand_yesterday_mean"] = d["day"].map(d.groupby("day")["demand"].mean().shift(1))
    for suffix in ("", "_yesterday"):
        temperature = d["temperature" + suffix]
        d["heating_degrees" + suffix] = (COMFORT_TEMPERATURE - temperature).clip(lower=0)
        d["cooling_degrees" + suffix] = (temperature - COMFORT_TEMPERATURE).clip(lower=0)
    return d


def fit_linear(X, y):
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)
    return coef


def predict_linear(coef, X):
    return coef[0] + X @ coef[1:]


def new_boosting(num_leaves, n_trees, learning_rate):
    return LGBMRegressor(num_leaves=num_leaves, n_estimators=n_trees, learning_rate=learning_rate,
                         min_child_samples=20, random_state=SEED, n_jobs=1, deterministic=True,
                         force_row_wise=True, verbose=-1)


def choose_settings(X_fit, y_fit, X_val, y_val):
    """Choose the settings of both tree-based models on the validation weeks.

    Boosting adds trees one after the other, so a model with 8,000 trees
    contains the model with 250 trees: we fit the largest one and read the
    forecasts of the smaller ones from it.
    """
    def validation_rmse(model, n_trees):
        return float(np.sqrt(np.mean((model.predict(X_val, num_iteration=n_trees) - y_val) ** 2)))

    largest = new_boosting(2, max(ADDITIVE_TREES), 0.1).fit(X_fit, y_fit)
    additive_scores = {n: validation_rmse(largest, n) for n in ADDITIVE_TREES}

    gbm_scores = {}
    for leaves in sorted({leaves for leaves, _ in GBM_GRID}):
        trees = [n for l, n in GBM_GRID if l == leaves]
        largest = new_boosting(leaves, max(trees), 0.05).fit(X_fit, y_fit)
        gbm_scores.update({(leaves, n): validation_rmse(largest, n) for n in trees})
    return (min(additive_scores, key=additive_scores.get), min(gbm_scores, key=gbm_scores.get),
            additive_scores, gbm_scores)


def block_bootstrap_days(n_days, rng):
    """Resample days in blocks of consecutive days, to respect the dependence between nearby days."""
    starts = rng.integers(0, n_days - BLOCK_DAYS + 1, size=int(np.ceil(n_days / BLOCK_DAYS)))
    return np.concatenate([np.arange(s, s + BLOCK_DAYS) for s in starts])[:n_days]


def main():
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    OUTPUT_DIR.mkdir(exist_ok=True)

    d = build_inputs(load_vic_electricity())
    d = d[d["day"] >= FIRST_USABLE_DAY].reset_index(drop=True)
    day = d["day"].to_numpy()
    y = d["demand"].to_numpy()
    X_tree, X_lin = d[TREE_INPUTS].to_numpy(float), d[LINEAR_INPUTS].to_numpy(float)
    n_test_days = int(day.max()) - FIRST_TEST_DAY + 1
    print(f"{len(d)} half-hours; test: the {n_test_days} days of 2014 ({n_test_days * PER_DAY} half-hours)")

    # ---- 1. Settings chosen once, before the test year ---------------------------------
    fit_part = day < FIRST_TEST_DAY - VALIDATION_DAYS
    val_part = (day >= FIRST_TEST_DAY - VALIDATION_DAYS) & (day < FIRST_TEST_DAY)
    additive_trees, (gbm_leaves, gbm_trees), additive_scores, gbm_scores = choose_settings(
        X_tree[fit_part], y[fit_part], X_tree[val_part], y[val_part])
    print(f"Chosen on the last {VALIDATION_DAYS} days of 2013: additive model with {additive_trees} trees; "
          f"gradient boosting with {gbm_trees} trees of {gbm_leaves} leaves")

    # ---- 2. Rolling origin: the models are refitted at the start of each month of 2014 ----
    month = (pd.Timestamp("2012-01-01") + pd.to_timedelta(day, unit="D")).month.to_numpy()
    test = day >= FIRST_TEST_DAY
    predictions = {m: np.full(len(y), np.nan) for m in METHODS}
    predictions[NAIVE][test] = d["demand_last_week"].to_numpy()[test]
    for m in range(1, 13):
        target = test & (month == m)
        past = day < day[target].min()                # everything before the month being forecast
        predictions[LINEAR][target] = predict_linear(fit_linear(X_lin[past], y[past]), X_lin[target])
        predictions[ADDITIVE][target] = new_boosting(2, additive_trees, 0.1).fit(
            X_tree[past], y[past]).predict(X_tree[target])
        predictions[GBM][target] = new_boosting(gbm_leaves, gbm_trees, 0.05).fit(
            X_tree[past], y[past]).predict(X_tree[target])

    forecasts = pd.DataFrame({"time": d["time"].dt.strftime("%Y-%m-%d %H:%M")[test], "day": day[test],
                              "temperature": d["temperature"][test], "actual": y[test]})
    forecasts = forecasts.reset_index(drop=True)
    for m in METHODS:
        forecasts[m] = predictions[m][test]
    forecasts.round(2).to_csv(OUTPUT_DIR / "e2b_forecasts.csv", index=False)

    # ---- 3. Errors with block-bootstrap intervals ----------------------------------------
    # MAPE: average error as a percentage of the actual demand. RMSE: in MWh.
    actual = forecasts["actual"].to_numpy().reshape(n_test_days, PER_DAY)
    squared, percent = {}, {}
    for m in METHODS:
        error = forecasts[m].to_numpy().reshape(n_test_days, PER_DAY) - actual
        squared[m] = (error ** 2).sum(axis=1)                    # one value per day
        percent[m] = (np.abs(error) / actual).sum(axis=1) * 100

    def rmse(m, days):
        return float(np.sqrt(squared[m][days].sum() / (len(days) * PER_DAY)))

    def mape(m, days):
        return float(percent[m][days].sum() / (len(days) * PER_DAY))

    all_days = np.arange(n_test_days)
    rng = np.random.default_rng(SEED)
    boot = {m: {"rmse": np.empty(N_BOOTSTRAP), "mape": np.empty(N_BOOTSTRAP)} for m in METHODS}
    for b in range(N_BOOTSTRAP):
        days = block_bootstrap_days(n_test_days, rng)
        for m in METHODS:
            boot[m]["rmse"][b], boot[m]["mape"][b] = rmse(m, days), mape(m, days)

    result_rows = []
    for m in METHODS:
        row = {"method": m}
        for name, function in (("mape", mape), ("rmse", rmse)):
            low, high = percentile_interval(boot[m][name])
            row.update({name: function(m, all_days), f"{name}_low": low, f"{name}_high": high})
        result_rows.append(row)
    results = pd.DataFrame(result_rows)

    gap_rows = []
    for first, second in ((GBM, ADDITIVE), (GBM, LINEAR), (ADDITIVE, LINEAR), (GBM, NAIVE)):
        row = {"comparison": f"{first} minus {second}"}
        for name, function in (("mape", mape), ("rmse", rmse)):
            low, high = percentile_interval(boot[first][name] - boot[second][name])
            row.update({f"{name}_gap": function(first, all_days) - function(second, all_days),
                        f"{name}_gap_low": low, f"{name}_gap_high": high})
        gap_rows.append(row)
    gaps = pd.DataFrame(gap_rows)
    results.round(4).to_csv(OUTPUT_DIR / "e2b_results.csv", index=False)
    gaps.round(4).to_csv(OUTPUT_DIR / "e2b_gaps.csv", index=False)

    # Errors on the peaks: the half-hours with the highest demand of the year,
    # which are the moments that matter most to a grid operator. This table was
    # added after looking at the forecasts of the hottest week (see the figure).
    n_peaks = int(round(PEAK_SHARE * len(forecasts)))
    peak_rows = forecasts.nlargest(n_peaks, "actual")
    peaks = pd.DataFrame([{
        "method": m,
        "mape": float((np.abs(peak_rows[m] - peak_rows["actual"]) / peak_rows["actual"]).mean() * 100),
        "mean_error_percent": float(((peak_rows[m] - peak_rows["actual"]) / peak_rows["actual"]).mean() * 100),
    } for m in METHODS])
    peaks.round(4).to_csv(OUTPUT_DIR / "e2b_peaks.csv", index=False)

    # ---- 4. The models in readable form (fitted on all the data) ---------------------------
    coef = fit_linear(X_lin, y)
    lines = ["== Linear regression ==", f"demand = {coef[0]:+.1f}"]
    lines += [f"    {c:+.3f} x {name}" for name, c in zip(LINEAR_INPUTS, coef[1:])]

    additive = new_boosting(2, additive_trees, 0.1).fit(X_tree, y)
    # For an additive model, the contribution of an input depends on that input
    # only. We read the curve of each input from the contributions on the data.
    contributions = additive.predict(X_tree, pred_contrib=True)
    curve_rows = []
    for j, name in enumerate(TREE_INPUTS):
        values = X_tree[:, j]
        if name.startswith("temperature"):
            values = np.round(values)              # one point per degree
        elif name.startswith("demand"):
            values = np.round(values, -2)          # one point per 100 MWh
        curve = pd.DataFrame({"value": values, "effect": contributions[:, j]}).groupby("value")["effect"]
        curve_rows.append(pd.DataFrame({"input": name, "value": curve.mean().index,
                                        "effect_on_demand": curve.mean().to_numpy(),
                                        "half_hours": curve.size().to_numpy()}))
    curves = pd.concat(curve_rows, ignore_index=True)
    curves.round(2).to_csv(OUTPUT_DIR / "e2b_curves.csv", index=False)
    spread = curves.groupby("input")["effect_on_demand"].agg(lambda e: e.max() - e.min())
    lines += ["", "== Additive model ==",
              f"sum of {additive_trees} trees with two leaves: one curve per input (see e2b_curves.csv)",
              "range of each curve (highest effect minus lowest effect on demand):"]
    lines += [f"    {spread[name]:8.1f} for {name}" for name in sorted(TREE_INPUTS, key=lambda n: -spread[n])]
    lines += ["validation RMSE of the six candidates: "
              + ", ".join(f"{n} trees: {s:.1f}" for n, s in additive_scores.items()),
              "", "== Gradient boosting ==",
              f"sum of {gbm_trees} trees with {gbm_leaves} leaves each: not readable directly",
              "validation RMSE of the six candidates: "
              + ", ".join(f"{n} trees of {l} leaves: {s:.1f}" for (l, n), s in gbm_scores.items())]
    (OUTPUT_DIR / "e2b_models.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\nForecast errors over 2014 (95% block-bootstrap intervals)")
    print(results.round(2).to_string(index=False))
    print("\nGaps (negative = the first method is better)")
    print(gaps.round(2).to_string(index=False))
    print(f"\nErrors on the {n_peaks} half-hours with the highest demand "
          "(mean error below zero = demand is underestimated)")
    print(peaks.round(2).to_string(index=False))
    print("\n" + "\n".join(lines))

    make_figure(forecasts, results, curves)
    print(f"\nFiles written in {OUTPUT_DIR}")


def make_figure(forecasts, results, curves):
    colors = {NAIVE: COLOR_REFERENCE, LINEAR: COLOR_INTERPRETABLE, ADDITIVE: "#7fbfa9", GBM: COLOR_BLACK_BOX}
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1.5, 1, 1]})

    # Left: the week that contains the highest demand of the year.
    ax = axes[0]
    peak_day = int(forecasts.loc[forecasts["actual"].idxmax(), "day"])
    week = forecasts[forecasts["day"].between(peak_day - 3, peak_day + 3)]
    dates = pd.to_datetime(week["time"])
    ax.plot(dates, week["actual"], color="#c0392b", linewidth=2, label="actual")
    ax.plot(dates, week[LINEAR], color=colors[LINEAR], linewidth=1.2, label=LINEAR)
    ax.plot(dates, week[GBM], "--", color=colors[GBM], linewidth=1.2, label=GBM)
    ax.set_ylabel("electricity demand (MWh)")
    ax.set_title("Week of the highest demand of 2014", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # Middle: average error over the whole year.
    ax = axes[1]
    part = results.set_index("method").loc[METHODS]
    x = np.arange(len(METHODS))
    ax.bar(x, part["mape"], color=[colors[m] for m in METHODS],
           yerr=[part["mape"] - part["mape_low"], part["mape_high"] - part["mape"]],
           error_kw={"capsize": 3, "ecolor": "#666666"})
    for xi, value in zip(x, part["mape"]):
        ax.text(xi, 0.1, f"{value:.1f}%", ha="center", va="bottom", color="white", fontsize=9)
    ax.set_xticks(x, ["Naive", "Linear", "Additive", "Gradient\nboosting"])
    ax.set_ylabel("average error (% of demand)")
    ax.set_title("Error over 2014", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: what the additive model learned about temperature.
    ax = axes[2]
    curve = curves[(curves["input"] == "temperature") & (curves["half_hours"] >= 20)]
    ax.plot(curve["value"], curve["effect_on_demand"], color=colors[ADDITIVE], linewidth=2)
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_xlabel("temperature at the forecast time (°C)")
    ax.set_ylabel("effect on demand (MWh)")
    ax.set_title("Additive model: the temperature curve", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "e2b_figure.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
