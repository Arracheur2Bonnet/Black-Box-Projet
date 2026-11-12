# Black boxes, explainable AI, interpretable AI

Code for a course project on the debate opened by Rudin (2019), "Stop explaining
black box machine learning models for high stakes decisions and use interpretable
models instead".

## The question

Does an interpretable model cost accuracy, and can a post-hoc explanation of a
black box be trusted? We test this on a high-stakes classification problem and on
a forecasting problem.

Two terms are used throughout:

- An **interpretable** model can be read directly, for example a short list of
  rules or a linear model with a few coefficients.
- An **explainable** model is a black box that comes with a second tool (such as
  SHAP) which gives an approximate explanation of its predictions.

In this project a model counts as interpretable if it uses at most 10 variables
(linear model) or has a depth of at most 3 (decision tree).

## The three experiments

| | Question | Data | Models |
|---|---|---|---|
| E1 | What does interpretability cost in accuracy? | COMPAS recidivism data (ProPublica) | Sparse logistic regression, small decision tree, gradient boosting |
| E2 | Does the black box win in forecasting? | US unemployment rate 12 months ahead, FRED-MD panel of monthly macroeconomic series | Naive forecast, autoregressive model, sparse linear model, gradient boosting |
| E3 | Do equally accurate models give the same explanation? | Same as E1 | 50 gradient boosting models with nearly the same accuracy |

The protocol is described in [docs/methodology.md](docs/methodology.md) and the
results in [docs/results.md](docs/results.md).

## Status

| Experiment | Script | State |
|---|---|---|
| E1 | `src/e1_compas.py` | done |
| E2 | `src/e2_forecast.py` | done |
| E3 | `src/e3_rashomon.py` | done |

Each script downloads its data if needed, prints its result tables and writes
its figure in `outputs/`. For example: `python src/e1_compas.py`.

## Setup

Python 3.11 is required.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS, Linux
pip install -r requirements.txt
```

## Repository layout

```
requirements.txt    pinned library versions
src/                one script per experiment, plus shared helpers
docs/               methodology and results
```

The data are not stored in this repository. The scripts download them into a
local `data/` folder and write figures and result files into a local `outputs/`
folder. Both folders are ignored by git.
