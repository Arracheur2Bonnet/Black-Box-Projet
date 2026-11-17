# Black boxes, explainable AI, interpretable AI

Code for a course project on the debate opened by Rudin (2019), "Stop explaining
black box machine learning models for high stakes decisions and use interpretable
models instead".

## The question

Does an interpretable model cost accuracy, and can a post-hoc explanation of a
black box be trusted? We measure this on a high-stakes classification problem
and on two forecasting problems.

Two terms are used throughout:

- An **interpretable** model can be read directly, for example a short list of
  rules or a linear model with a few coefficients.
- An **explainable** model is a black box that comes with a second tool (such as
  SHAP) which gives an approximate explanation of its predictions.

In this project a model counts as interpretable if it uses at most 10 variables
(linear model) or has a depth of at most 3 (decision tree).

## The experiments

| | Question | Data | Models | Script | Run time |
|---|---|---|---|---|---|
| E1 | What does interpretability cost in accuracy? | COMPAS recidivism data (ProPublica) | Sparse logistic regression, small decision tree, gradient boosting | `src/e1_compas.py` | 2 min |
| E2a | Does the black box win at forecasting unemployment? | US unemployment rate 12 months ahead, FRED-MD panel of monthly macroeconomic series | Naive forecast, autoregressive model, sparse linear model, gradient boosting | `src/e2a_unemployment.py` | 1 min |
| E2b | Does the black box win at forecasting electricity demand? | Half-hourly electricity demand of Victoria (Australia), one day ahead | Naive forecast, linear regression, additive model, the same two fitted per half-hour, gradient boosting | `src/e2b_electricity.py` | 10 min |
| E3 | Do equally accurate models give the same explanation and the same decision? | Same as E1 | 50 gradient boosting models with nearly the same accuracy; as a control, 50 logistic regressions | `src/e3_rashomon.py` | 1 min |

The protocol of each experiment and its limits are described in
[docs/methodology.md](docs/methodology.md). The result tables are in
[docs/results.md](docs/results.md).

## What we found

- **E1.** On the recidivism data, a logistic regression that uses 5 variables is
  as accurate as the gradient boosting (accuracy 0.683 for both). The black box
  is slightly ahead on finer measures (0.005 of AUC, 0.006 of Brier score). The
  rule list published by Rudin (2019) is as accurate as the COMPAS score (0.667
  against 0.661).
- **E2a.** For unemployment 12 months ahead, the black box does not win. The
  most accurate forecast comes from the simplest readable model, an
  autoregression on unemployment alone, the only method that beats the naive
  forecast.
- **E2b.** For electricity demand, the black box clearly beats the simplest
  readable models (2.9% of error against 5.1% for a linear regression with 9
  variables). Readable models fitted per half-hour, as in the load forecasting
  literature, come close: 3.2% for one linear regression per half-hour and 3.0%
  for an additive model per half-hour, whose gap with the black box is not
  distinguishable from zero.
- **E3.** Fifty equally accurate black boxes agree on which variables matter
  overall. For one person, they often disagree: for 23% of people they do not
  give the same yes/no prediction. Fifty equally accurate logistic regressions
  disagree almost as much (21%): the instability comes from the multiplicity of
  good models, not from the explanations.

Taken together, interpretability cost little in our tests once the readable
model was built with the same care as the black box; the black box found by
itself a structure that the readable model needed from domain knowledge. It is
for the black box to show its advantage, against a readable model built with
the same effort.

## Running the code

Python 3.11 is required.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS, Linux
pip install -r requirements.txt
python src/e1_compas.py
```

Each script downloads its data if needed, prints its result tables and writes
its figure and result files in `outputs/`. Once the four experiments have
run, `python src/report_figures.py` draws the figures of the written report
from these result files, in `outputs/report/`. The report uses the Archivo
typeface; to draw the figures with it, pass the folder that contains its font
files: `python src/report_figures.py --fonts <folder>`. Every random step uses
the seed set in `src/common.py`, so two runs give the same numbers.

Some models and comparisons were added after a review of the first results:
the two models per half-hour of E2b, the logistic-regression control of E3, the
Brier score of E1 and the comparison of the autoregression with the naive
forecast in E2a. They are marked as such in `docs/`.

## Repository layout

```
requirements.txt    pinned library versions
src/common.py       seed, folders, bootstrap helpers
src/data.py         download and preparation of the data sets
src/e*.py           one script per experiment
src/report_figures.py   figures of the written report
docs/               methodology and results
```

The data are not stored in this repository. The scripts download them into a
local `data/` folder and write figures and result files into a local `outputs/`
folder. Both folders are ignored by git.

## Data sources

- COMPAS: file `compas-scores-two-years.csv` published by ProPublica with
  "How We Analyzed the COMPAS Recidivism Algorithm" (2016), read at a fixed
  commit of their repository.
- FRED-MD: monthly database of the Federal Reserve Bank of St. Louis (McCracken
  and Ng, 2016), release of September 2026.
- Electricity demand of Victoria: data set `vic_elec` of the R package
  tsibbledata, used in Hyndman and Athanasopoulos, "Forecasting: Principles and
  Practice" (3rd edition), read from the CSV copy of the skforecast project.
