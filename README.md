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
| E2b | Does the black box win at forecasting electricity demand? | Half-hourly electricity demand of Victoria (Australia), one day ahead | Naive forecast, linear regression, additive model, gradient boosting | `src/e2b_electricity.py` | 8 min |
| E3 | Do equally accurate models give the same explanation? | Same as E1 | 50 gradient boosting models with nearly the same accuracy | `src/e3_rashomon.py` | 1 min |

The protocol of each experiment and its limits are described in
[docs/methodology.md](docs/methodology.md). The result tables are in
[docs/results.md](docs/results.md).

## What we found

- **E1.** On the recidivism data, a logistic regression that uses 5 variables is
  as accurate as the gradient boosting (accuracy 0.683 for both). The
  three-condition rule list published by Rudin (2019) is as accurate as the
  COMPAS score (0.667 against 0.661).
- **E2a.** For unemployment 12 months ahead, the black box does not win. No
  method is clearly better than the naive forecast, and the intervals are too
  wide to rank the methods.
- **E2b.** For electricity demand, the black box wins clearly: its average error
  is 2.9%, against 5.1% for a linear regression with 9 variables and 4.5% for
  an additive model that can be read curve by curve.
- **E3.** Fifty equally accurate black boxes agree on which variables matter
  overall. For one person, they often disagree: for 36% of people they do not
  name the same most important variable, and for 23% they do not give the same
  yes/no prediction.

Taken together, the cost of interpretability was zero on the recidivism data,
not measurable on the unemployment forecasts and real on the electricity
forecasts. It has to be measured case by case, and it is for the black box to
show its advantage.

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
from these result files. Every random step uses the seed set
in `src/common.py`, so two runs give the same numbers.

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
  "How We Analyzed the COMPAS Recidivism Algorithm" (2016).
- FRED-MD: monthly database of the Federal Reserve Bank of St. Louis (McCracken
  and Ng, 2016), release of September 2026.
- Electricity demand of Victoria: data set `vic_elec` of the R package
  tsibbledata, used in Hyndman and Athanasopoulos, "Forecasting: Principles and
  Practice" (3rd edition), read from the CSV copy of the skforecast project.
