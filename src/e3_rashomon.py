"""Experiment 3 (Rashomon): do equally accurate models give the same explanation?

We train many gradient boosting models on the same COMPAS training data with
different settings and random seeds, keep 50 whose test AUC is within 0.01 of
the best one, and explain each of them with SHAP. We then measure how much the
50 explanations agree, for the data set as a whole and for each person.

Control (added after review): the same measures for 200 sparse logistic
regressions instead of gradient boostings, drawn in two ways (each on a random
share of the rows, or all on every row with only the penalty varying). If
equally accurate readable models also disagree, the disagreement comes from
the multiplicity of good models (the Rashomon effect), not only from post-hoc
explanations or black boxes.

Run:  python src/e3_rashomon.py
Writes outputs/e3_summary.csv, e3_ranks.csv, e3_model_ranks.csv,
e3_model_importance.csv, e3_example.csv, e3_example_models.csv, e3_control.csv
and e3_figure.png.
"""

import itertools
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from scipy.stats import kendalltau, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (COLOR_BLACK_BOX, COLOR_GUIDE, COLOR_INTERPRETABLE, LIGHTGBM_REPRODUCIBLE, N_BOOTSTRAP,
                    OUTPUT_DIR, SEED, bootstrap_indices)
from data import COMPAS_FEATURES, load_compas

N_CANDIDATES = 200      # models trained
N_KEPT = 50             # models kept among those that are nearly as good as the best
AUC_TOLERANCE = 0.01    # "nearly as good": test AUC within 0.01 of the best
TEST_SHARE = 0.30
DECISION_THRESHOLD = 0.5   # a predicted risk of 0.5 or more means "re-arrest"
NEAR_THRESHOLD = 0.1       # a risk within 0.1 of the threshold is a borderline case

# The first agreement measure. The control table also uses it to find the people
# whose yes/no prediction changes.
FLIP_ANY = "share of people whose yes/no prediction differs between models"


def random_settings(rng, seed):
    """Draw one gradient boosting configuration at random.

    The draws are made in the order of the arguments below. With the generator
    of `main`, this order fixes the 200 candidates: reordering the arguments
    would give other models.
    """
    return dict(
        num_leaves=int(rng.choice([2, 3, 4, 6, 8, 12, 16])),
        n_estimators=int(rng.choice([50, 100, 200, 300, 400])),
        learning_rate=float(rng.choice([0.02, 0.05, 0.1])),
        subsample=float(rng.choice([0.5, 0.6, 0.7, 0.8, 0.9])),   # rows used by each tree
        subsample_freq=1,
        colsample_bytree=float(rng.choice([0.4, 0.6, 0.8, 1.0])),  # variables used by each tree
        min_child_samples=int(rng.choice([10, 25, 50, 100])),
        random_state=seed, **LIGHTGBM_REPRODUCIBLE,
    )


def shap_values(model, X):
    """SHAP value of each variable for each person (in log-odds of re-arrest).

    For one person, the SHAP values say how much each variable pushed the
    prediction of this model above or below its average prediction.
    """
    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):      # older SHAP versions return one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def scored(model, X_test, y_test):
    """A fitted candidate in the form in which candidates are kept: (test AUC, model, predicted risks)."""
    proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, proba), model, proba


def keep_equally_accurate(candidates, rng):
    """Draw N_KEPT models at random among the candidates within AUC_TOLERANCE of the best.

    The models are drawn at random among the good ones, not taken as the top
    N_KEPT, so that nothing but "nearly as accurate as the best" decides which
    models are compared. Returns the test AUC of every candidate, the indices
    of the good ones and the sorted indices of the kept ones.
    """
    aucs = np.array([c[0] for c in candidates])
    good = np.flatnonzero(aucs >= aucs.max() - AUC_TOLERANCE)
    kept = np.sort(rng.choice(good, size=min(N_KEPT, len(good)), replace=False))
    return aucs, good, kept


def train_boostings(rng, X_train, y_train, X_test, y_test):
    """Train the N_CANDIDATES gradient boostings, each with settings drawn from `rng` and its own seed."""
    candidates = []
    for i in range(N_CANDIDATES):
        model = LGBMClassifier(**random_settings(rng, seed=SEED + i)).fit(X_train, y_train)
        candidates.append(scored(model, X_test, y_test))
    return candidates


def logistic_control(X_train, y_train, X_test, y_test, subsample):
    """50 equally accurate sparse logistic regressions, drawn like the 50 gradient boostings.

    Each candidate has its own penalty (from 0.003 to 10, on a log scale). With
    `subsample`, each candidate is also trained on a random 50 to 90% of the
    training rows, the share that each tree of a boosting sees; a boosting as a
    whole, however, sees every row. Without it, every candidate sees all the
    rows and only the penalty varies. For a linear model, the contribution of a
    variable to one prediction is exactly its coefficient times the gap between
    the person's value and the average value, which is also its SHAP value.
    Returns the predicted risks and the main variable of each person in each
    kept model.
    """
    rng = np.random.default_rng(SEED + 1)      # separate from the draws of the boostings
    candidates = []
    for _ in range(N_CANDIDATES):
        if subsample:
            rows = rng.choice(len(X_train), size=int(len(X_train) * rng.uniform(0.5, 0.9)), replace=False)
        else:
            rows = np.arange(len(X_train))
        penalty = float(np.exp(rng.uniform(np.log(0.003), np.log(10))))
        model = make_pipeline(StandardScaler(), LogisticRegression(C=penalty, l1_ratio=1.0, solver="liblinear", random_state=SEED))
        model.fit(X_train.iloc[rows], y_train.iloc[rows])
        candidates.append(scored(model, X_test, y_test))
    aucs, good, kept = keep_equally_accurate(candidates, rng)
    probas = np.array([candidates[i][2] for i in kept])
    top = []
    for i in kept:
        scaler, logit = candidates[i][1].named_steps["standardscaler"], candidates[i][1].named_steps["logisticregression"]
        contributions = (X_test.to_numpy() - scaler.mean_) / scaler.scale_ * logit.coef_[0]
        top.append(np.abs(contributions).argmax(axis=1))
    return probas, np.array(top), len(good), aucs[kept]


def agreement_measures(probas, top):
    """One value per person for each measure; the measure itself is the average over people.

    The first measure counts a person as soon as one model disagrees with the
    others, so it grows with the number of models compared. The second one
    does not: for each person, it is the share of pairs of models that give
    different answers, so its average is the share of people whose answer
    differs between two models drawn at random.
    """
    decisions = probas >= DECISION_THRESHOLD
    n = decisions.shape[0]
    flips = decisions.any(axis=0) & ~decisions.all(axis=0)
    yes = decisions.sum(axis=0)
    pair_flips = yes * (n - yes) / (n * (n - 1) / 2)
    same_all = np.array([len(np.unique(top[:, j])) == 1 for j in range(top.shape[1])])
    pair_share = np.array([(np.bincount(top[:, j]).astype(float) * (np.bincount(top[:, j]) - 1)).sum()
                           / (n * (n - 1)) for j in range(top.shape[1])])
    return {FLIP_ANY: flips,
            "average share of people whose yes/no prediction differs between two models": pair_flips,
            "share of people with the same main variable in all models": same_all,
            "average share of people with the same main variable in two models": pair_share}


def explain_all(models, X_test):
    """SHAP values of every kept model, and what they say about each variable.

    Returns the SHAP values (models, people, variables), the importance of each
    variable in each model (mean absolute SHAP value, in log-odds), the rank of
    each variable in each model, and the main variable of each person in each
    model (the one with the largest absolute SHAP value).
    """
    shap_all = np.array([shap_values(m, X_test) for m in models])  # (models, people, variables)
    importance = np.abs(shap_all).mean(axis=1)                     # (models, variables)
    # Rank 1 = most important. A variable that a model never uses has an
    # importance of exactly zero; such variables share the last rank.
    ranks = rankdata(-importance, axis=1, method="max").astype(int)
    top = np.abs(shap_all).argmax(axis=2)                          # (models, people)
    return shap_all, importance, ranks, top


def summary_table(n_good, kept_aucs, n_test, probas, shap_all, importance, ranks, top, pairs):
    """How much the kept models agree: in their global ranking, for each person, and in their predictions."""
    # Global explanation: two models agree when they rank the variables alike.
    taus = np.array([kendalltau(importance[a], importance[b]).statistic for a, b in pairs])
    # Local explanation: the main variable of each person. `same_top` is the same
    # quantity as the pairwise measure of agreement_measures, computed per pair of
    # models instead of per person.
    n_distinct_top = np.array([len(np.unique(top[:, j])) for j in range(n_test)])
    same_top = np.array([(top[a] == top[b]).mean() for a, b in pairs])
    # Does a variable clearly push the risk up in one model and clearly down in
    # another, for the same person? Effects below 0.1 in absolute value are
    # ignored: values close to zero change sign for no meaningful reason.
    sign_conflict = ((shap_all > 0.1).any(axis=0) & (shap_all < -0.1).any(axis=0))  # (people, variables)
    # Predictions themselves: do equally accurate models give the same person the same risk?
    spread = probas.max(axis=0) - probas.min(axis=0)
    decisions = probas >= DECISION_THRESHOLD
    flips = decisions.any(axis=0) & ~decisions.all(axis=0)

    summary = pd.DataFrame([
        ("models trained", N_CANDIDATES),
        ("models within 0.01 AUC of the best", n_good),
        ("models kept", len(kept_aucs)),
        ("lowest AUC among kept models", kept_aucs.min()),
        ("highest AUC among kept models", kept_aucs.max()),
        ("people in the test set", n_test),
        ("global ranking: median Kendall tau between two models", np.median(taus)),
        ("global ranking: lowest Kendall tau between two models", taus.min()),
        ("global ranking: share of model pairs with identical ranking", (taus > 0.999).mean()),
        ("global ranking: number of different variables ranked first", len(np.unique(ranks.argmin(axis=1)))),
        ("local: average share of people with the same top variable in two models", same_top.mean()),
        ("local: share of people with the same top variable in all models", (n_distinct_top == 1).mean()),
        ("local: share of people with a variable whose effect clearly changes sign between models",
         sign_conflict.any(axis=1).mean()),
        ("prediction: median gap between highest and lowest predicted risk", np.median(spread)),
        ("prediction: share of people whose yes/no prediction differs between models", flips.mean()),
    ], columns=["quantity", "value"])
    summary["value"] = summary["value"].astype(float).round(4)
    return summary


def central_95(values):
    """2.5% and 97.5% quantiles of resampled values.

    The quantiles 0.025 and 0.975 are written out: common.percentile_interval
    computes (1 - 0.95) / 2, which is not exactly 0.025 in floating point and
    would change the last digits of the saved intervals.
    """
    return np.quantile(values, 0.025), np.quantile(values, 0.975)


def control_table(families, n_test):
    """The agreement measures of each family of 50 models, with 95% intervals over people.

    `families` maps a name to (predicted risks, main variable of each person),
    both of shape (models, people). The 1,852 people of the test set are
    resampled N_BOOTSTRAP times, and every family is measured on the same
    resamples.
    """
    people = bootstrap_indices(n_test, N_BOOTSTRAP, seed=SEED + 2)
    rows = []
    for family, (family_probas, family_top) in families.items():
        measures = agreement_measures(family_probas, family_top)
        for measure, per_person in measures.items():
            low, high = central_95(per_person[people].mean(axis=1))
            rows.append({"family": family, "measure": measure, "value": per_person.mean(),
                         "low": low, "high": high})
        # Are the people whose answer changes the borderline cases? Among them,
        # the share whose risk, averaged over the 50 models, is from 0.4 to 0.6.
        flips = measures[FLIP_ANY]
        near = np.abs(family_probas.mean(axis=0) - DECISION_THRESHOLD) <= NEAR_THRESHOLD
        low, high = central_95((flips & near)[people].sum(axis=1) / flips[people].sum(axis=1))
        rows.append({"family": family, "value": (flips & near).sum() / flips.sum(),
                     "measure": "among these people, share whose average risk is from 0.4 to 0.6",
                     "low": low, "high": high})
    control = pd.DataFrame(rows)
    control["value"] = control["value"].astype(float)
    return control


def rank_counts(ranks):
    """Number of models that give each rank to each variable (one row per variable)."""
    table = pd.DataFrame(
        {f"rank {r}": (ranks == r).sum(axis=0) for r in range(1, ranks.shape[1] + 1)},
        index=COMPAS_FEATURES)
    table.index.name = "variable"
    return table


def per_model_table(values, kept_aucs):
    """One row per kept model: its number, its test AUC and one value per variable."""
    table = pd.DataFrame(values, columns=COMPAS_FEATURES)
    table.insert(0, "auc", kept_aucs.round(4))
    table.insert(0, "model", np.arange(1, len(values) + 1))
    return table


def pick_example(pairs, top, shap_all):
    """One person explained differently by two models: the most affirmative case found.

    Among the people whose main variable differs between two models, take the
    case where both models are the most affirmative about their own main
    variable. Returns the two models and the person.
    """
    best = None
    for a, b in pairs:
        differ = np.flatnonzero(top[a] != top[b])
        if len(differ) == 0:
            continue
        strength = np.minimum(np.abs(shap_all[a, differ]).max(axis=1), np.abs(shap_all[b, differ]).max(axis=1))
        j = differ[strength.argmax()]
        if best is None or strength.max() > best[0]:
            best = (strength.max(), a, b, j)
    _, a, b, j = best
    return a, b, j


def example_tables(a, b, j, X_test, shap_all, kept_aucs, probas):
    """The example person: the SHAP values given by the two models, and their AUC and predicted risk."""
    example = pd.DataFrame({
        "variable": COMPAS_FEATURES,
        "value for this person": X_test.iloc[j].to_numpy(),
        "SHAP in model A": shap_all[a, j].round(3),
        "SHAP in model B": shap_all[b, j].round(3),
    })
    two_models = pd.DataFrame({"model": ["A", "B"], "auc": [round(kept_aucs[a], 3), round(kept_aucs[b], 3)],
                               "predicted_risk": [round(probas[a, j], 2), round(probas[b, j], 2)]})
    return example, two_models


def main():
    warnings.filterwarnings("ignore")   # hide library warnings, so that the printed tables stay readable
    OUTPUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    data = load_compas()
    X, y = data[COMPAS_FEATURES], data["two_year_recid"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SHARE, stratify=y, random_state=SEED)
    n_test = X_test.shape[0]

    # ---- 1. Train many models and keep 50 that are nearly as good as the best ----
    # `rng` serves these two steps only, in this order.
    candidates = train_boostings(rng, X_train, y_train, X_test, y_test)
    aucs, good, kept = keep_equally_accurate(candidates, rng)
    models = [candidates[i][1] for i in kept]
    probas = np.array([candidates[i][2] for i in kept])           # (models, people)
    kept_aucs = aucs[kept]
    print(f"{len(good)} of {N_CANDIDATES} models are within {AUC_TOLERANCE} AUC of the best "
          f"({aucs.max():.3f}); {len(kept)} kept, AUC from {kept_aucs.min():.3f} to {kept_aucs.max():.3f}")

    # ---- 2. Explain every model with SHAP and measure how much they agree ----------
    shap_all, importance, ranks, top = explain_all(models, X_test)
    pairs = list(itertools.combinations(range(len(models)), 2))
    summary = summary_table(len(good), kept_aucs, n_test, probas, shap_all, importance, ranks, top, pairs)
    summary.to_csv(OUTPUT_DIR / "e3_summary.csv", index=False)

    # Control: the same measures for 50 equally accurate logistic regressions, in two
    # versions: each regression trained on a random 50 to 90% of the rows, or all of
    # them trained on every row with only the penalty varying.
    families = {"gradient boosting": (probas, top)}
    for family, subsample in (("sparse logistic regression, each on 50-90% of rows", True),
                              ("sparse logistic regression, all rows", False)):
        control_probas, control_top, control_good, control_aucs = logistic_control(
            X_train, y_train, X_test, y_test, subsample)
        families[family] = (control_probas, control_top)
        print(f"\nControl ({family}): {control_good} of {N_CANDIDATES} within {AUC_TOLERANCE} AUC of the best; "
              f"{len(control_aucs)} kept, AUC from {control_aucs.min():.3f} to {control_aucs.max():.3f}")
    control = control_table(families, n_test)
    control.round(4).to_csv(OUTPUT_DIR / "e3_control.csv", index=False)
    print(control.round(3).to_string(index=False))

    # The rank of each variable in each model, and the mean absolute SHAP value
    # behind it (log-odds), which shows how much more the first variables weigh.
    rank_table = rank_counts(ranks)
    rank_table.to_csv(OUTPUT_DIR / "e3_ranks.csv")
    per_model_table(ranks, kept_aucs).to_csv(OUTPUT_DIR / "e3_model_ranks.csv", index=False)
    per_model_table(importance.round(4), kept_aucs).to_csv(OUTPUT_DIR / "e3_model_importance.csv", index=False)

    # ---- 3. One person, two models, two explanations ------------------------------
    a, b, j = pick_example(pairs, top, shap_all)
    example, two_models = example_tables(a, b, j, X_test, shap_all, kept_aucs, probas)
    example.to_csv(OUTPUT_DIR / "e3_example.csv", index=False)
    two_models.to_csv(OUTPUT_DIR / "e3_example_models.csv", index=False)
    example_note = (f"model A: AUC {kept_aucs[a]:.3f}, predicted risk {probas[a, j]:.2f}; "
                    f"model B: AUC {kept_aucs[b]:.3f}, predicted risk {probas[b, j]:.2f}")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 100)
    print("\n" + summary.to_string(index=False))
    print("\nNumber of models (out of 50) giving each rank to each variable")
    print(rank_table.to_string())
    print("\nOne person, two models: " + example_note)
    print(example.to_string(index=False))

    make_figure(rank_table, example, kept_aucs[a], kept_aucs[b])
    print(f"\nFiles written in {OUTPUT_DIR}")


def make_figure(rank_table, example, auc_a, auc_b):
    """Quick-look figure of E3: the ranks given by the 50 models, and the example person.

    The report uses figure 4, drawn by report_figures.py, instead.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    # Left: how the 50 models rank each variable.
    ax = axes[0]
    order = rank_table.mul(np.arange(1, rank_table.shape[1] + 1), axis=1).sum(axis=1).sort_values().index
    table = rank_table.loc[order]
    ax.imshow(table.to_numpy(), cmap="Greens", vmin=0, vmax=table.to_numpy().max(), aspect="auto")
    for i in range(table.shape[0]):
        for k in range(table.shape[1]):
            count = table.iat[i, k]
            if count:
                ax.text(k, i, str(count), ha="center", va="center", fontsize=9,
                        color="white" if count > table.to_numpy().max() / 2 else "black")
    ax.set_xticks(range(table.shape[1]), [str(r) for r in range(1, table.shape[1] + 1)])
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_xlabel("importance rank given by SHAP (1 = most important)")
    ax.set_title("Number of models giving each rank to each variable", fontsize=10)

    # Right: the same person explained by two models of the same accuracy.
    ax = axes[1]
    positions = np.arange(len(example))[::-1]
    ax.barh(positions + 0.2, example["SHAP in model A"], height=0.38, color=COLOR_BLACK_BOX,
            label=f"model A (AUC {auc_a:.3f})")
    ax.barh(positions - 0.2, example["SHAP in model B"], height=0.38, color=COLOR_INTERPRETABLE,
            label=f"model B (AUC {auc_b:.3f})")
    labels = [f"{v} = {x:g}" for v, x in zip(example["variable"], example["value for this person"])]
    ax.set_yticks(positions, labels)
    ax.axvline(0, color=COLOR_GUIDE, linewidth=0.8)
    ax.set_xlabel("SHAP value (effect on the log-odds of re-arrest)")
    ax.set_title("One person explained by two models", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "e3_figure.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
