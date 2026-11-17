# Results

All numbers are produced by the scripts in `src/` with the seed set in
`src/common.py`. Intervals are 95% bootstrap intervals. Some models and
comparisons were added after a review of the first results; they are marked as
such below.

## E1: what does interpretability cost in accuracy?

COMPAS data, 6,172 people. Run time: about 2 minutes.

| Method | Kind | AUC | Accuracy | Brier score |
|---|---|---|---|---|
| Sparse logistic regression | interpretable | 0.731 [0.718, 0.744] | 0.683 [0.672, 0.695] | 0.212 [0.209, 0.216] |
| Decision tree (depth at most 3) | interpretable | 0.708 [0.694, 0.721] | 0.670 [0.658, 0.682] | 0.214 [0.209, 0.218] |
| Gradient boosting | black box | 0.736 [0.723, 0.748] | 0.683 [0.671, 0.694] | 0.206 [0.202, 0.211] |
| COMPAS score | reference | 0.710 [0.698, 0.722] | 0.661 [0.649, 0.673] | |
| Rudin (2019) rule list | reference | not comparable | 0.667 [0.655, 0.679] | |
| Always predict "no re-arrest" | baseline | | 0.545 | |

The Brier score is the mean squared gap between the predicted probability and
the outcome (lower is better). It applies only to the methods that give a
probability.

Gap between the gradient boosting and each other method:

| Gradient boosting minus | AUC gap | Accuracy gap | Brier gap |
|---|---|---|---|
| Sparse logistic regression | 0.005 [0.001, 0.009] | -0.001 [-0.007, 0.005] | -0.006 [-0.008, -0.004] |
| Decision tree (depth at most 3) | 0.028 [0.022, 0.034] | 0.013 [0.005, 0.021] | -0.007 [-0.009, -0.005] |
| COMPAS score | 0.026 [0.014, 0.037] | 0.022 [0.009, 0.034] | |
| Rudin (2019) rule list | not comparable | 0.016 [0.006, 0.025] | |

A positive AUC or accuracy gap, and a negative Brier gap, mean that the black
box is better.

What the numbers show:

- The black box and the sparse logistic regression have the same accuracy. On
  finer measures the black box is slightly ahead (0.005 of AUC, 0.006 of Brier
  score): the gaps are real but too small to change a decision about a person.
- The published rule list is as accurate as the COMPAS score on our data (0.667
  against 0.661). It was learned by others on Broward County data that overlap
  ours, so its accuracy is partly measured on the data it was learned from.
- The small decision tree is behind, so an interpretable model is not
  automatically as good as a black box: its form matters.
- No method goes much above 0.68 in accuracy. With these inputs, re-arrest is
  hard to predict for every model.

The fitted interpretable models can be read in full. The logistic regression
uses 5 of the 7 inputs:

```
log-odds of re-arrest = 0.432
    - 0.035 x age
    + 0.165 x male
    + 0.140 x number of priors
    + 0.141 x number of other juvenile offenses
    + 0.113 x felony charge
```

In 8 of the 10 folds, the setting chosen for the gradient boosting was 300 trees
with two leaves each (100 trees with four leaves in the other 2). A tree with
two leaves uses one variable, so such a model adds one curve per variable and
combines none: it is an additive model, of the kind counted as readable in E2b.
The data did not reward combining variables.

## E2a: does the black box win at forecasting unemployment?

Change in the US unemployment rate over 12 months, 429 forecasts from January
1990 to September 2025. Errors are root mean squared errors (RMSE) in
percentage points of unemployment. Run time: about 1 minute.

| Method | Kind | RMSE, all months | RMSE, without Covid period |
|---|---|---|---|
| Naive (no change) | baseline | 1.52 [0.73, 2.30] | 0.96 [0.66, 1.33] |
| Autoregressive model | interpretable | 1.82 [0.63, 3.00] | 0.82 [0.58, 1.12] |
| Sparse linear model | interpretable | 1.53 [0.70, 2.33] | 0.93 [0.60, 1.32] |
| Gradient boosting | black box | 1.51 [0.76, 2.23] | 0.97 [0.64, 1.22] |

Gap in RMSE without the Covid period (negative means the first method is
better):

| Comparison | RMSE gap |
|---|---|
| Gradient boosting minus sparse linear model | 0.04 [-0.18, 0.25] |
| Gradient boosting minus autoregressive model | 0.15 [-0.03, 0.30] |
| Gradient boosting minus naive | 0.01 [-0.19, 0.20] |
| Sparse linear model minus naive | -0.03 [-0.11, 0.03] |
| Autoregressive model minus naive (added after review) | -0.14 [-0.23, -0.04] |

What the numbers show:

- The black box does not win. Outside the Covid period its error (0.97) is no
  lower than that of the sparse linear model (0.93).
- The most accurate forecast comes from the simplest readable model, the
  autoregression on the unemployment rate alone (0.82). It is the only method
  that beats the naive forecast: the interval of its gap excludes zero.
- Forecasting unemployment a year ahead is hard for every model: the forecasts
  follow the turning points with a delay instead of announcing them.
- With all months included, the Covid jump of April 2020 dominates the error
  and the intervals become too wide to compare anything.
- From 2023 to 2025, the forecasts of the gradient boosting stay well above the
  observed rate, while the linear models come back to it, most likely because
  the extreme values of 2020 entered its training data.

The interpretable models can be read in full. Fitted on all the data, the sparse
linear model uses 5 of the 120 inputs. The effect of a one-standard-deviation
rise of each input on the forecast change is:

```
change over the next 12 months = 0.473
    - 0.126  unemployment rate (level)
    - 0.107  5-year Treasury rate minus federal funds rate
    - 0.085  real M2 money stock (growth)
    - 0.057  1-year Treasury rate minus federal funds rate
    - 0.056  3-month Treasury bill rate minus federal funds rate
```

In words: unemployment tends to fall when it is high, and to rise after
long-term rates have fallen below the policy rate, a known recession signal.
The gradient boosting gives its largest SHAP values to the same two inputs (the
5-year rate spread and the unemployment level), followed by employment series.
It is a sum of 300 trees over 41 inputs and cannot be read directly.

## E2b: does the black box win at forecasting electricity demand?

Half-hourly electricity demand of Victoria (Australia), forecast one day ahead
over the year 2014: 17,520 half-hours. MAPE is the average error as a
percentage of the actual demand. Run time: about 10 minutes.

| Method | Kind | MAPE | RMSE (MWh) |
|---|---|---|---|
| Naive (same time last week) | baseline | 7.1% [5.7, 8.7] | 613 [428, 801] |
| Linear regression, 9 variables | interpretable | 5.1% [4.6, 5.5] | 322 [283, 367] |
| Additive model | readable curve by curve | 4.5% [4.1, 4.9] | 277 [243, 310] |
| Linear regression per half-hour (added after review) | interpretable | 3.2% [2.9, 3.5] | 219 [189, 246] |
| Additive model per half-hour (added after review) | readable map by map | 3.0% [2.6, 3.3] | 208 [170, 244] |
| Gradient boosting | black box | 2.9% [2.6, 3.2] | 208 [172, 241] |

Gap in MAPE, in percentage points (negative means the first method is better):

| Comparison | MAPE gap |
|---|---|
| Gradient boosting minus additive model | -1.6 [-1.7, -1.4] |
| Gradient boosting minus linear regression | -2.1 [-2.4, -1.9] |
| Additive model minus linear regression | -0.6 [-0.8, -0.4] |
| Gradient boosting minus linear regression per half-hour | -0.3 [-0.4, -0.1] |
| Gradient boosting minus additive model per half-hour | -0.1 [-0.2, 0.1] |

What the numbers show:

- Against the readable models we first used, the black box wins clearly: its
  error is about 40% lower than that of the linear regression. These two models
  cannot let an effect change with the time of day, and this is where the black
  box gains.
- Readable models that can do it come close. One linear regression per
  half-hour, with the same 9 variables, is 0.3 point behind the black box; an
  additive model whose curves may change with the half-hour is 0.08 point
  behind, and that gap is not distinguishable from zero.
- The cost of interpretability therefore depends on the structure given to the
  readable model: about two points for the simplest, almost nothing for those
  built as in the load forecasting literature (Fan and Hyndman, 2012). The
  black box found this structure by itself; the readable models needed it from
  domain knowledge.

The readable models can be read in full. The single linear regression is:

```
demand = 232
    + 543   if the day is a working day
    - 389   if the day before was a working day
    + 49    per heating degree (degrees below 18 °C)
    + 91    per cooling degree (degrees above 18 °C)
    - 23    per heating degree at the same time the day before
    - 43    per cooling degree at the same time the day before
    + 0.66  x demand at the same time the day before
    + 0.26  x demand at the same time one week before
    - 0.03  x average demand of the day before
```

The regression per half-hour has the same form, with one set of coefficients
for each half-hour (`outputs/e2b_halfhour_coefficients.csv`). For example, one
degree above 18 °C adds about 36 MWh of demand at 4 a.m. and 112 MWh at 5 p.m.

The temperature curve of the additive model has the expected U shape, and it is
not symmetric. Demand is lowest around 19 °C. Compared with that point, it is
about 440 MWh higher at 3 °C and about 3,000 MWh higher at 41 °C. The gradient
boosting is a sum of 1,000 trees with 8 leaves each and cannot be read directly.

Errors on the 175 half-hours with the highest demand of 2014 (11 days, mostly
during the heat wave of 13 to 17 January). This table was added after looking at
the forecasts of the hottest week:

| Method | MAPE | Mean error |
|---|---|---|
| Naive (same time last week) | 33.5% | -33.5% |
| Linear regression | 9.5% | -9.0% |
| Additive model | 9.0% | -7.4% |
| Linear regression per half-hour | 7.0% | -6.8% |
| Additive model per half-hour | 7.8% | -6.9% |
| Gradient boosting | 7.5% | -6.6% |

- Every model underestimates the peaks, and the regression per half-hour is the
  most accurate there.
- The highest demand of 2014 (9,345 MWh) is above every value of 2012 and 2013
  (at most 8,897 MWh). The highest forecast of the gradient boosting over the
  whole year is 8,540 MWh: a model made of trees does not forecast values
  outside the range it has seen. An average error therefore hides how a model
  behaves in the situations that matter most, which have to be checked
  separately.

## E3: do equally accurate models give the same explanation?

COMPAS data, 1,852 people in the test set. Run time: about 1 minute.

Of the 200 gradient boostings trained, 144 are within 0.01 AUC of the best one
(0.728). The 50 models kept have a test AUC between 0.719 and 0.728.

| Level | Measure | Value |
|---|---|---|
| Whole data set | Median Kendall tau between the SHAP rankings of two models | 0.90 |
| Whole data set | Lowest Kendall tau between two models | 0.62 |
| Whole data set | Variables ranked first by at least one model | 1 (number of priors) |
| One person | People with the same main variable in two models (average over pairs) | 92% |
| One person | People with the same main variable in all 50 models | 64% |
| One person | People with a variable whose effect clearly changes sign between models | 11% |
| Prediction | Median gap between the highest and lowest risk given to one person | 0.17 |
| Prediction | People whose yes/no prediction differs between models | 23% |

Number of models, out of 50, that give each importance rank to each variable:

| Variable | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Number of priors | 50 | | | | | | |
| Age | | 50 | | | | | |
| Felony charge | | | 44 | 5 | | | 1 |
| Male | | | 3 | 41 | 4 | | 2 |
| Other juvenile offenses | | | 3 | 3 | 44 | | |
| Juvenile misdemeanors | | | | | | 26 | 24 |
| Juvenile felonies | | | | | | 19 | 31 |

A variable that a model does not use at all is ranked last. This happens in 15
of the 50 models, mostly for the number of juvenile felonies.

Control (added after review): the same measures for 50 equally accurate sparse
logistic regressions (198 of 200 were within 0.01 AUC of the best one).

| Measure | Gradient boostings | Logistic regressions |
|---|---|---|
| People whose yes/no prediction differs between models | 23% [21, 25] | 21% [19, 23] |
| People with the same main variable in all 50 models | 64% [62, 66] | 68% [66, 70] |
| People with the same main variable in two models | 92% [91, 92] | 92% [91, 93] |

What the numbers show:

- For the data set as a whole, the 50 models tell the same story: the number of
  priors comes first and age second in every model.
- For one person, the story is less stable, and so is the decision: for 23% of
  people, some of these equally accurate models predict re-arrest and others do
  not.
- Equally accurate logistic regressions disagree almost as much. The
  instability comes from the multiplicity of good models (the Rashomon effect),
  not from post-hoc explanations or from black boxes. An explanation describes
  one model, not what another equally good model would decide.

One example from the test set: a 61-year-old man with 20 priors and a
non-felony charge. Model A (AUC 0.727) gives him a risk of 0.65 and its largest
SHAP value is the number of priors (+1.94 against -1.00 for age). Model B (AUC
0.721) gives him a risk of 0.34 and its largest SHAP value is age (-1.33 against
+1.18 for priors). He is the most extreme disagreement found by searching all
pairs of models and all people: he illustrates the effect, he does not measure
it.
