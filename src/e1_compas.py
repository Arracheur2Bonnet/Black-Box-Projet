"""Experiment 1: what does interpretability cost in accuracy on COMPAS?

We compare, on the same people and with the same protocol:
  - two interpretable models (sparse logistic regression, small decision tree),
  - one black box (gradient boosting),
  - two references that are not trained here: the COMPAS score and the
    three-rule list published in Rudin (2019), plus the "always predict no
    re-arrest" baseline for accuracy.

Run:  python src/e1_compas.py
Writes outputs/e1_results.csv, e1_gaps.csv, e1_models.txt and e1_figure.png.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from common import (COLOR_BLACK_BOX, COLOR_INTERPRETABLE, COLOR_REFERENCE, OUTPUT_DIR,
                    SEED, bootstrap_indices, percentile_interval)
from data import COMPAS_FEATURES, load_compas

N_OUTER_FOLDS = 10     # each person is predicted once, by a model that never saw them
N_INNER_FOLDS = 5      # used inside the training data to choose the settings
N_BOOTSTRAP = 2000

BLACK_BOX = "Gradient boosting"


def candidate_models():
    """The three trained models, each with a grid of six candidate settings.

    The grids have the same size so that no model gets more tuning effort than
    another. The gradient boosting grid includes very simple versions (trees
    with two leaves), because on about 6,000 rows a large boosting model
    overfits and would make the black box look worse than it can be.
    """
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(l1_ratio=1.0, solver="liblinear", random_state=SEED),
    )
    logistic_grid = {"logisticregression__C": [0.003, 0.01, 0.03, 0.1, 1.0, 10.0]}

    tree = DecisionTreeClassifier(random_state=SEED)
    tree_grid = {"max_depth": [1, 2, 3], "min_samples_leaf": [25, 100]}

    boosting = LGBMClassifier(learning_rate=0.05, min_child_samples=25, subsample=1.0,
                              random_state=SEED, n_jobs=1, deterministic=True,
                              force_row_wise=True, verbose=-1)
    boosting_grid = {"num_leaves": [2, 4, 8], "n_estimators": [100, 300]}

    return {
        "Sparse logistic regression": ("interpretable", logistic, logistic_grid),
        "Decision tree (depth <= 3)": ("interpretable", tree, tree_grid),
        BLACK_BOX: ("black box", boosting, boosting_grid),
    }


def rudin_rule(data):
    """The rule list of Rudin (2019), Figure 3, applied as published.

    IF age 18-20 and male THEN predict arrest
    ELSE IF age 21-23 and 2-3 prior offenses THEN predict arrest
    ELSE IF more than three priors THEN predict arrest
    ELSE predict no arrest.
    """
    age, priors = data["age"], data["priors_count"]
    arrest = (
        (age.between(18, 20) & (data["male"] == 1))
        | (age.between(21, 23) & priors.between(2, 3))
        | (priors > 3)
    )
    return arrest.astype(int).to_numpy()


def out_of_fold_scores(model, grid, X, y):
    """Predicted probability of re-arrest for every person, by nested cross-validation.

    Outer loop: the data are split in 10 parts; each part is predicted by a
    model trained on the other 9. Inner loop: within those 9 parts, a 5-fold
    search picks the setting with the best AUC. The test part is never used to
    choose a setting.

    Returns the scores and the setting chosen in each of the 10 outer folds.
    """
    outer = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED + 1)
    scores, chosen = np.zeros(len(y)), []
    for train, test in outer.split(X, y):
        search = GridSearchCV(model, grid, scoring="roc_auc", cv=inner)
        search.fit(X.iloc[train], y.iloc[train])
        scores[test] = search.predict_proba(X.iloc[test])[:, 1]
        chosen.append(str(search.best_params_))
    return scores, chosen


def describe_final_models(models, X, y, chosen_in_folds):
    """Refit each model on all the data and write it down in readable form."""
    inner = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED + 1)
    lines = []
    for name, (_, model, grid) in models.items():
        search = GridSearchCV(model, grid, scoring="roc_auc", cv=inner).fit(X, y)
        best = search.best_estimator_
        lines.append(f"== {name} ==")
        lines.append(f"chosen settings: {search.best_params_}")
        counts = pd.Series(chosen_in_folds[name]).value_counts()
        lines.append("settings chosen in the 10 outer folds: "
                     + ", ".join(f"{setting}: {n} times" for setting, n in counts.items()))
        if name.startswith("Sparse"):
            # Coefficients are converted back to the original units, so that
            # "+0.15 per prior offense" can be read directly.
            scaler, logit = best.named_steps["standardscaler"], best.named_steps["logisticregression"]
            coefs = logit.coef_[0] / scaler.scale_
            intercept = logit.intercept_[0] - np.sum(coefs * scaler.mean_)
            lines.append("log-odds of re-arrest = " + f"{intercept:+.3f}")
            for feature, coef in zip(COMPAS_FEATURES, coefs):
                lines.append(f"    {coef:+.3f} x {feature}" if coef != 0 else f"    (unused) {feature}")
            lines.append(f"variables used: {int(np.sum(coefs != 0))} of {len(COMPAS_FEATURES)}")
        elif name.startswith("Decision"):
            lines.append(export_text(best, feature_names=COMPAS_FEATURES, show_weights=False))
            lines.append(f"number of leaves: {best.get_n_leaves()}")
        else:
            n_trees = best.n_estimators
            lines.append(f"sum of {n_trees} trees with {best.num_leaves} leaves each: no short formula")
            if best.num_leaves == 2:
                lines.append("a tree with two leaves uses one variable, so this model adds one curve per variable "
                             "and combines none: it is an additive model, of the kind counted as readable in E2b")
        lines.append("")
    return "\n".join(lines)


def main():
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = load_compas()
    X, y = data[COMPAS_FEATURES], data["two_year_recid"]
    y_np = y.to_numpy()
    print(f"{len(data)} people, {y.mean():.1%} re-arrested within two years")

    # ---- 1. Scores of every method for every person -------------------------
    models = candidate_models()
    kinds, scores, labels, chosen = {}, {}, {}, {}
    for name, (kind, model, grid) in models.items():
        print(f"Cross-validating: {name}")
        kinds[name] = kind
        scores[name], chosen[name] = out_of_fold_scores(model, grid, X, y)
        labels[name] = (scores[name] >= 0.5).astype(int)
    trained = list(models)   # methods that give a probability: the Brier score applies to them only

    # References that are not trained on these data.
    name = "COMPAS score"
    kinds[name] = "reference"
    scores[name] = data["decile_score"].to_numpy().astype(float)
    labels[name] = (data["decile_score"] >= 5).to_numpy().astype(int)  # "medium" or "high" risk

    name = "Rudin (2019) rule list"
    kinds[name] = "reference"
    labels[name] = rudin_rule(data)
    scores[name] = labels[name].astype(float)  # a rule gives yes/no, not a score

    # ---- 2. Accuracy, AUC and Brier score with bootstrap intervals ----------
    # The Brier score is the mean squared gap between the predicted probability
    # and the outcome (0 or 1); lower is better. It does not apply to the COMPAS
    # score (1 to 10) nor to the rule list (yes or no), which give no probability.
    idx = bootstrap_indices(len(y_np), N_BOOTSTRAP)
    boot = {name: {"auc": np.empty(N_BOOTSTRAP), "accuracy": np.empty(N_BOOTSTRAP),
                   "brier": np.full(N_BOOTSTRAP, np.nan)} for name in scores}
    for b, rows in enumerate(idx):
        for name in scores:
            boot[name]["auc"][b] = roc_auc_score(y_np[rows], scores[name][rows])
            boot[name]["accuracy"][b] = accuracy_score(y_np[rows], labels[name][rows])
            if name in trained:
                boot[name]["brier"][b] = brier_score_loss(y_np[rows], scores[name][rows])

    rows = []
    for name in scores:
        auc_low, auc_high = percentile_interval(boot[name]["auc"])
        acc_low, acc_high = percentile_interval(boot[name]["accuracy"])
        if name in trained:
            brier = brier_score_loss(y_np, scores[name])
            brier_low, brier_high = percentile_interval(boot[name]["brier"])
        else:
            brier = brier_low = brier_high = np.nan
        rows.append({
            "model": name, "kind": kinds[name],
            "auc": roc_auc_score(y_np, scores[name]), "auc_low": auc_low, "auc_high": auc_high,
            "accuracy": accuracy_score(y_np, labels[name]), "accuracy_low": acc_low, "accuracy_high": acc_high,
            "brier": brier, "brier_low": brier_low, "brier_high": brier_high,
        })
    results = pd.DataFrame(rows)
    baseline_accuracy = max(y.mean(), 1 - y.mean())  # always predict the most frequent outcome

    # ---- 3. Gap between the black box and every other method ----------------
    # AUC and accuracy gaps: positive = the black box is better.
    # Brier gap: negative = the black box is better (trained methods only).
    gap_rows = []
    for name in scores:
        if name == BLACK_BOX:
            continue
        row = {"comparison": f"{BLACK_BOX} minus {name}"}
        for metric in ("auc", "accuracy", "brier"):
            if metric == "brier" and name not in trained:
                row.update({"brier_gap": np.nan, "brier_gap_low": np.nan, "brier_gap_high": np.nan})
                continue
            differences = boot[BLACK_BOX][metric] - boot[name][metric]
            point = (results.set_index("model").loc[BLACK_BOX, metric]
                     - results.set_index("model").loc[name, metric])
            low, high = percentile_interval(differences)
            row.update({f"{metric}_gap": point, f"{metric}_gap_low": low, f"{metric}_gap_high": high})
        gap_rows.append(row)
    gaps = pd.DataFrame(gap_rows)

    # ---- 4. Save and print ---------------------------------------------------
    results.round(6).to_csv(OUTPUT_DIR / "e1_results.csv", index=False)
    gaps.round(6).to_csv(OUTPUT_DIR / "e1_gaps.csv", index=False)
    (OUTPUT_DIR / "e1_models.txt").write_text(describe_final_models(models, X, y, chosen), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\nResults (95% bootstrap intervals)")
    print(results.round(3).to_string(index=False))
    print(f"\nBaseline accuracy (always predict no re-arrest): {baseline_accuracy:.3f}")
    print("\nGaps (AUC, accuracy: positive = the black box is better; Brier: negative = the black box is better)")
    print(gaps.round(3).to_string(index=False))

    make_figure(results, baseline_accuracy)
    print(f"\nFiles written in {OUTPUT_DIR}")


def make_figure(results, baseline_accuracy):
    colors = {"interpretable": COLOR_INTERPRETABLE, "black box": COLOR_BLACK_BOX, "reference": COLOR_REFERENCE}
    order = results.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, metric, title in zip(axes, ("auc", "accuracy"), ("AUC", "Accuracy")):
        values = order[metric]
        errors = [values - order[f"{metric}_low"], order[f"{metric}_high"] - values]
        ax.barh(order["model"], values, xerr=errors, color=[colors[k] for k in order["kind"]],
                error_kw={"capsize": 3, "ecolor": "#666666"})
        for y_pos, value in enumerate(values):
            ax.text(0.505, y_pos, f"{value:.3f}", va="center", color="white", fontsize=9)
        ax.set_xlim(0.5, 0.78)
        ax.set_title(f"{title} with 95% bootstrap interval", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].axvline(baseline_accuracy, color="#c0392b", linestyle="--", linewidth=1)
    axes[1].set_ylim(-0.6, len(order) + 0.1)
    axes[1].text(baseline_accuracy + 0.004, len(order) - 0.45, "always predict no re-arrest",
                 color="#c0392b", fontsize=8, va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "e1_figure.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
