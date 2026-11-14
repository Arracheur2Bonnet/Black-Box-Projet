"""Figures of the written report, drawn from the result files of the experiments.

This script runs no experiment. It reads the files that the experiment scripts
wrote in `outputs/` and draws the four figures in one common style.

Run:  python src/report_figures.py [--fonts FOLDER]
Writes outputs/report/*.pdf (and .png). With --fonts, the Archivo typeface found
in FOLDER is used; otherwise the default sans-serif font.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

from common import OUTPUT_DIR

# One colour per kind of model, the same in every figure.
INTERPRETABLE = "#FA520F"   # orange
ADDITIVE = "#FDA47E"        # light orange: readable curve by curve
BLACK_BOX = "#151515"       # black
REFERENCE = "#BDB4A0"       # warm grey: baselines and references
MUTE = "#6B6B6B"            # grey for axes and notes
TEXT_WIDTH = 6.3            # inches: width of the text block of the report (16 cm)

REPORT_DIR = OUTPUT_DIR / "report"

VARIABLE_NAMES = {"priors_count": "prior offenses", "age": "age", "felony_charge": "felony charge",
                  "male": "sex", "juv_other_count": "other juvenile offenses",
                  "juv_misd_count": "juvenile misdemeanors", "juv_fel_count": "juvenile felonies"}


def set_style(font_folder):
    if font_folder:
        for path in Path(font_folder).glob("Archivo-*.ttf"):
            font_manager.fontManager.addfont(str(path))
        plt.rcParams["font.family"] = "Archivo"
    plt.rcParams.update({
        "font.size": 8.5, "axes.edgecolor": MUTE, "axes.labelcolor": MUTE, "axes.titlesize": 8.5,
        "axes.titlecolor": BLACK_BOX, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": MUTE, "ytick.color": BLACK_BOX, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save(fig, name):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_DIR / f"{name}.pdf")
    fig.savefig(REPORT_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


def colour_key(ax, entries, y=1.04, gap=0.025):
    """Small colour key above or below an axis: (colour, label, horizontal position)."""
    for color, label, x in entries:
        ax.plot([x + 0.008], [y + 0.035], marker="s", markersize=5, color=color, transform=ax.transAxes, clip_on=False)
        ax.text(x + gap, y, label, color=MUTE, transform=ax.transAxes, fontsize=8, va="bottom")


def bars_with_intervals(ax, rows, start, number_format):
    """Horizontal bars from `start`, with the 95% interval and the value written at the end.

    rows: list of (label, colour, value, low, high), from top to bottom.
    """
    for position, (label, color, value, low, high) in enumerate(reversed(rows)):
        ax.barh(position, value - start, left=start, color=color, height=0.62)
        # 95% interval: drawn in white over the bar and in grey beyond it.
        ax.plot([low, value], [position, position], color="white", linewidth=1.1)
        ax.plot([value, high], [position, position], color=MUTE, linewidth=1.1)
        ax.text(high, position, "  " + number_format.format(value), va="center", fontsize=8.5,
                fontweight="bold", color=BLACK_BOX)
    ax.set_yticks(range(len(rows)), [row[0] for row in reversed(rows)])


def figure_e1():
    """Accuracy of the five methods on the COMPAS data."""
    results = pd.read_csv(OUTPUT_DIR / "e1_results.csv").set_index("model")
    spec = [("Sparse logistic regression", "Logistic regression, 5 variables", INTERPRETABLE),
            ("Decision tree (depth <= 3)", "Decision tree of depth 3", INTERPRETABLE),
            ("Gradient boosting", "Gradient boosting", BLACK_BOX),
            ("Rudin (2019) rule list", "Rudin's three rules", REFERENCE),
            ("COMPAS score", "COMPAS score", REFERENCE)]
    rows = [(label, color, *results.loc[key, ["accuracy", "accuracy_low", "accuracy_high"]]) for key, label, color in spec]
    baseline = 0.545   # accuracy of always predicting "no new arrest"

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.0))
    bars_with_intervals(ax, rows, baseline, "{:.3f}")
    ax.set_xlim(baseline, 0.715)
    ax.set_xticks([baseline, 0.58, 0.62, 0.66, 0.70], ["0.545\nalways “no”", "0.58", "0.62", "0.66", "0.70"])
    ax.set_xlabel("accuracy on people the model has not seen")
    colour_key(ax, ((INTERPRETABLE, "interpretable", 0.0), (BLACK_BOX, "black box", 0.2),
                    (REFERENCE, "reference, not trained here", 0.36)))
    fig.tight_layout(pad=0.4)
    save(fig, "fig1_e1_accuracy")


def figure_e2a():
    """US unemployment rate and the forecasts made twelve months earlier."""
    f = pd.read_csv(OUTPUT_DIR / "e2a_forecasts.csv")
    dates = pd.PeriodIndex(f["month"], freq="M").to_timestamp()

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 1.9))
    for start, end in (("1990-07", "1991-03"), ("2001-03", "2001-11"), ("2007-12", "2009-06")):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#F1EEE6", linewidth=0)
    ax.axvspan(pd.Timestamp("2020-03"), pd.Timestamp("2021-12"), color="#F1EEE6", linewidth=0)
    ax.plot(dates, f["actual"], color=REFERENCE, linewidth=2.6, label="actual rate")
    ax.plot(dates, f["Sparse linear model"], color=INTERPRETABLE, linewidth=0.95, label="sparse linear model (interpretable)")
    ax.plot(dates, f["Gradient boosting"], color=BLACK_BOX, linewidth=0.95, label="gradient boosting (black box)")
    ax.set_ylim(2.5, 12)
    ax.set_xlim(dates[0], dates[-1])
    ax.set_ylabel("unemployment rate (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=3, bbox_to_anchor=(-0.01, 1.2), handlelength=1.4,
              columnspacing=1.4, labelcolor=MUTE)
    fig.tight_layout(pad=0.4)
    save(fig, "fig2_e2a_unemployment")


def figure_e2b():
    """Forecast error on electricity demand, and the temperature curve of the additive model."""
    results = pd.read_csv(OUTPUT_DIR / "e2b_results.csv").set_index("method")
    spec = [("Naive (same time last week)", "Naive forecast", REFERENCE),
            ("Linear regression", "Linear regression, 9 variables", INTERPRETABLE),
            ("Additive model", "Additive model", ADDITIVE),
            ("Gradient boosting", "Gradient boosting", BLACK_BOX)]
    rows = [(label, color, *results.loc[key, ["mape", "mape_low", "mape_high"]]) for key, label, color in spec]
    curves = pd.read_csv(OUTPUT_DIR / "e2b_curves.csv")
    curve = curves[(curves["input"] == "temperature") & (curves["half_hours"] >= 20)]

    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.0), gridspec_kw={"width_ratios": [1.55, 1]})
    bars_with_intervals(left, rows, 0, "{:.1f}%")
    left.set_xlim(0, 10.6)
    left.set_xticks([0, 2, 4, 6, 8], ["0%", "2%", "4%", "6%", "8%"])
    left.set_xlabel("average forecast error (lower is better)")
    left.set_title("Error over the year 2014")

    right.plot(curve["value"], curve["effect_on_demand"], color=ADDITIVE, linewidth=2)
    right.axhline(0, color=MUTE, linewidth=0.5)
    lowest = curve.loc[curve["effect_on_demand"].idxmin()]
    right.annotate(f"lowest near {lowest['value']:.0f} °C", (lowest["value"], lowest["effect_on_demand"]),
                   xytext=(lowest["value"] - 3, 1500), fontsize=7.5, color=MUTE, ha="center",
                   arrowprops={"arrowstyle": "-", "color": MUTE, "linewidth": 0.6})
    right.set_xlabel("temperature (°C)")
    right.set_ylabel("effect on demand (MWh)")
    right.tick_params(axis="y", colors=MUTE)
    right.set_title("Effect of temperature")
    fig.tight_layout(pad=0.4, w_pad=1.6)
    save(fig, "fig3_e2b_electricity")


def figure_e3():
    """Ranks given by the 50 models to each variable, and one person explained by two models."""
    ranks = pd.read_csv(OUTPUT_DIR / "e3_ranks.csv", index_col=0)
    order = ["priors_count", "age", "felony_charge", "male", "juv_other_count", "juv_misd_count", "juv_fel_count"]
    example = pd.read_csv(OUTPUT_DIR / "e3_example.csv").set_index("variable")
    two_models = pd.read_csv(OUTPUT_DIR / "e3_example_models.csv").set_index("model")

    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.15), gridspec_kw={"width_ratios": [1.25, 1]})
    for row, variable in enumerate(order):
        for rank in range(1, 8):
            count = ranks.loc[variable, f"rank {rank}"]
            if count:
                shade = 0.12 + 0.88 * count / 50
                left.add_patch(plt.Rectangle((rank - 0.46, row - 0.4), 0.92, 0.8, color=BLACK_BOX, alpha=shade, linewidth=0))
                left.text(rank, row, str(count), ha="center", va="center", fontsize=7.5,
                          color="white" if shade > 0.5 else BLACK_BOX)
    left.set_xlim(0.5, 7.5)
    left.set_ylim(len(order) - 0.5, -0.5)
    left.set_xticks(range(1, 8))
    left.set_yticks(range(len(order)), [VARIABLE_NAMES[v] for v in order])
    left.set_xlabel("importance rank (1 = most important)")
    left.set_title("Number of models, out of 50, giving each rank")
    left.spines[["left", "bottom"]].set_visible(False)
    left.tick_params(axis="x", length=0)

    shown = ["priors_count", "age", "felony_charge"]
    for position, variable in enumerate(reversed(shown)):
        right.barh(position + 0.19, example.loc[variable, "SHAP in model A"], height=0.36, color=BLACK_BOX)
        right.barh(position - 0.19, example.loc[variable, "SHAP in model B"], height=0.36, color=REFERENCE)
    labels = [f"{VARIABLE_NAMES[v]} = {example.loc[v, 'value for this person']:g}".replace("felony charge = 0", "no felony charge")
              for v in reversed(shown)]
    right.set_yticks(range(len(shown)), labels)
    right.axvline(0, color=MUTE, linewidth=0.6)
    right.set_xlabel("pushes the risk down   |   up")
    right.set_title("One man, two models")
    right.tick_params(axis="x", colors=MUTE)
    colour_key(right, ((BLACK_BOX, f"model A: risk {two_models.loc['A', 'predicted_risk']:.2f}", -0.42),
                       (REFERENCE, f"model B: risk {two_models.loc['B', 'predicted_risk']:.2f}", 0.28)), y=-0.52, gap=0.05)
    fig.tight_layout(pad=0.4, w_pad=1.4)
    fig.subplots_adjust(bottom=0.3)
    save(fig, "fig4_e3_explanations")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fonts", help="folder that contains the Archivo font files")
    args = parser.parse_args()
    set_style(args.fonts)
    figure_e1()
    figure_e2a()
    figure_e2b()
    figure_e3()
    print(f"Figures written in {REPORT_DIR}")


if __name__ == "__main__":
    main()
