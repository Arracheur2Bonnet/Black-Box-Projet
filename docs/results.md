# Results

All numbers are produced by the scripts in `src/` with the seed set in
`src/common.py`. Intervals are 95% bootstrap intervals. Some models and
comparisons were added after a review of the first results; they are marked as
such below.

## E1: what does interpretability cost in accuracy?

COMPAS data, 6,172 people. Run time: about 2.5 minutes.

| Method | Kind | AUC | Accuracy | Brier score |
|---|---|---|---|---|
| Sparse logistic regression | interpretable | 0.731 [0.718, 0.744] | 0.674 [0.663, 0.686] | 0.212 [0.208, 0.215] |
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
| Sparse logistic regression | 0.005 [0.001, 0.009] | 0.008 [0.001, 0.016] | -0.006 [-0.007, -0.004] |
| Decision tree (depth at most 3) | 0.028 [0.022, 0.034] | 0.013 [0.005, 0.021] | -0.007 [-0.009, -0.005] |
| COMPAS score | 0.026 [0.014, 0.037] | 0.022 [0.009, 0.034] | |
| Rudin (2019) rule list | not comparable | 0.016 [0.006, 0.025] | |

A positive AUC or accuracy gap, and a negative Brier gap, mean that the black
box is better.

People whose yes/no prediction differs from that of the gradient boosting
(added after review). The last column gives, among these people, the share
for whom both predicted risks are between 0.4 and 0.6, that is, close to the
threshold of 0.5:

| Gradient boosting vs | People | Share of the 6,172 | Both risks between 0.4 and 0.6 |
|---|---|---|---|
| Sparse logistic regression | 506 | 8.2% [7.5, 8.8] | 80% |
| Decision tree (depth at most 3) | 663 | 10.7% [10.0, 11.5] | 58% |
| COMPAS score (5 or more) | 1,581 | 25.6% [24.5, 26.7] | |
| Rudin (2019) rule list | 849 | 13.8% [12.9, 14.6] | |

What the numbers show:

- The black box is slightly more accurate than the sparse logistic regression:
  0.8 point of accuracy (interval 0.1 to 1.6), 0.005 of AUC and 0.006 of Brier
  score. The cost of interpretability is small but measurable. The two models
  give a different yes/no prediction to 506 of the 6,172 people (8.2%); for 80%
  of them, both predicted risks are between 0.4 and 0.6. The average gap
  between the two predicted risks is 0.065.
- Corrected after review: the logistic regression is fitted with liblinear,
  which penalises the intercept like a coefficient. With the strong penalty
  chosen here, this shifted every predicted risk upwards (average 0.473 for a
  re-arrest rate of 0.455) and happened to give the same accuracy as the black
  box (0.683). The intercept is now left almost free (`intercept_scaling`);
  AUC and Brier score barely change, the accuracy falls to 0.674.
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
log-odds of re-arrest = 0.378
    - 0.036 x age
    + 0.168 x male
    + 0.138 x number of priors
    + 0.137 x number of other juvenile offenses
    + 0.116 x felony charge
```

In 8 of the 10 folds, the setting chosen for the gradient boosting was 300 trees
with two leaves each (100 trees with four leaves in the other 2). A tree with
two leaves uses one variable, so such a model adds one curve per variable and
combines none: it is an additive model, of the kind counted as readable in E2b.
The data did not reward combining variables.

## E2a: does the black box win at forecasting unemployment?

Change in the US unemployment rate over 12 months, 429 forecasts from January
1990 to September 2025. Errors are root mean squared errors (RMSE) in
percentage points of unemployment. The Covid period is made of the forecasts
for March 2020 to December 2021. Run time: about 2.5 minutes.

| Method | Kind | RMSE, all months | RMSE, without Covid period |
|---|---|---|---|
| Naive (no change) | baseline | 1.52 [0.73, 2.30] | 0.96 [0.66, 1.33] |
| Autoregressive model | interpretable | 1.82 [0.63, 3.00] | 0.82 [0.58, 1.12] |
| Sparse linear model | interpretable | 1.53 [0.70, 2.33] | 0.93 [0.60, 1.32] |
| Gradient boosting | black box | 1.51 [0.76, 2.23] | 0.97 [0.64, 1.22] |

Gap in RMSE (negative means the first method is better):

| Comparison | Without Covid period | All months |
|---|---|---|
| Gradient boosting minus sparse linear model | 0.04 [-0.18, 0.25] | -0.02 [-0.17, 0.15] |
| Gradient boosting minus autoregressive model | 0.15 [-0.03, 0.30] | -0.32 [-0.88, 0.21] |
| Gradient boosting minus naive | 0.01 [-0.19, 0.20] | -0.02 [-0.16, 0.13] |
| Sparse linear model minus autoregressive model | 0.11 [-0.02, 0.22] | -0.29 [-0.78, 0.19] |
| Sparse linear model minus naive | -0.03 [-0.11, 0.03] | 0.00 [-0.08, 0.06] |
| Autoregressive model minus naive (added after review) | -0.14 [-0.23, -0.04] | 0.30 [-0.20, 0.82] |

What the numbers show:

- The black box does not beat the readable models. Outside the Covid period its
  error (0.97) is no lower than that of the sparse linear model (0.93) or of the
  autoregression (0.82).
- Outside the Covid period, the lowest error comes from the simplest readable
  model, the autoregression on the unemployment rate alone (0.82). It is the
  only method whose gap with the naive forecast excludes zero. Its lead over
  the sparse linear model (0.11) and the gradient boosting (0.15) is not
  significant: both intervals contain zero.
- With the Covid period included, the ranking reverses: the autoregression has
  the largest error (1.82, against 1.51 for the gradient boosting). In April
  2020, with unemployment at 14.8%, it carried the jump forward and forecast
  26.1% for April 2021 (actual: 6.1%); the other two models also overshot, to
  17.1% and 15.2%. No gap can then be told apart from zero. The comparison
  without the Covid period was fixed in the protocol before the forecasts were
  run; it still removes the months in which the autoregression fails most.
- Forecasting unemployment a year ahead is hard for every model: the forecasts
  follow the turning points with a delay instead of announcing them.
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
  readable model: about two points for the simplest; 0.3 point, or 10% of the
  error, for the regression per half-hour, each of whose 48 equations meets our
  criterion of at most 10 variables; nothing measurable for the additive model
  per half-hour, which has no short formula and goes beyond the criterion. The
  models per half-hour follow the load forecasting literature (Fan and Hyndman,
  2012). The black box found this structure by itself; the readable models
  needed it from domain knowledge.

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
the forecasts of the hottest week. It has no interval and covers a single heat
wave, so it describes these days and does not test a difference. The
half-hours are selected on the actual demand, which by itself pushes the errors
downwards (corrected after review): selected on each model's own forecasts,
the mean errors are +0.2% for the linear regression, -3.7% for the regression
per half-hour and -3.3% for the gradient boosting, and the additive model per
half-hour has the lowest MAPE (5.7%). The table therefore does not rank the
models:

| Method | MAPE | Mean error |
|---|---|---|
| Naive (same time last week) | 33.5% | -33.5% |
| Linear regression | 9.5% | -9.0% |
| Additive model | 9.0% | -7.4% |
| Linear regression per half-hour | 7.0% | -6.8% |
| Additive model per half-hour | 7.8% | -6.9% |
| Gradient boosting | 7.5% | -6.6% |

- Every model underestimates these half-hours, partly because of how they are
  selected; the ranking of the models changes with the selection.
- The highest demand of 2014 (9,345 MWh) is above every value of 2012 and 2013
  (at most 8,897 MWh). The highest forecast of the gradient boosting over the
  whole year is 8,540 MWh: a model made of trees does not forecast values
  outside the range it has seen. The two linear regressions go above 9,500 MWh. An average error therefore hides how a model
  behaves in the situations that matter most, which have to be checked
  separately.

## E3: do equally accurate models give the same explanation?

COMPAS data, 1,852 people in the test set. Run time: about 1 minute.

Two of the measures below count people whose yes/no prediction differs. The
first counts a person as soon as one of the 50 models disagrees with the
others, so it grows with the number of models compared. The second is the
share of people whose prediction differs between two models drawn at random.

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
| Prediction | People whose yes/no prediction differs in at least one of the 50 models | 23% |
| Prediction | People whose yes/no prediction differs between two models (average over pairs) | 4.1% |

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

The ranks hide how far apart the variables are. Measured by the mean absolute
SHAP value, age, the second variable, weighs about five times as much as the
third one in the median model, and twice as much in the model where they are
closest. None of the five other variables exceeds 0.12 in any model, and the
first two make up 83% of the total importance in the median model. In model 1,
for example: priors 0.69, age 0.45, felony charge 0.09, sex 0.07, other
juvenile offenses 0.06, juvenile felonies 0.02, juvenile misdemeanors 0.01
(`outputs/e3_model_importance.csv`).

Control (added after review): the same measures for 50 equally accurate sparse
logistic regressions, drawn in two ways. In the first, each regression is
trained on a random 50 to 90% of the training rows, the share that each tree of
a boosting sees (199 of 200 candidates were within 0.01 AUC of the best one).
In the second, every regression sees all the rows and only the penalty varies
(200 of 200 within 0.01 AUC). As for the rows they see, the boostings sit in
between: each of their trees sees part of the rows, but the model as a whole
sees all of them. As in E1, the intercept of the regressions is left almost free
(corrected after review; with a penalised intercept the control gave 21% and
12% on the first line).

| Measure | Gradient boostings | Logistic regressions, each on 50-90% of rows | Logistic regressions, all rows |
|---|---|---|---|
| People whose yes/no prediction differs in at least one of the 50 models | 23.0% [21.1, 24.9] | 25.1% [23.1, 26.9] | 15.6% [13.9, 17.3] |
| People whose yes/no prediction differs between two models | 4.1% [3.7, 4.6] | 6.1% [5.5, 6.7] | 4.2% [3.7, 4.7] |
| Among the people of the first line, those whose average risk is between 0.4 and 0.6 | 80% [76, 84] | 88% [85, 91] | 93% [90, 96] |
| People with the same main variable in all 50 models | 64% [62, 66] | 65% [63, 67] | 82% [80, 83] |
| People with the same main variable in two models | 92% [91, 92] | 92% [92, 93] | 96% [96, 97] |

Sensitivity (added after review): the share of 23% depends on what "equally
good" means. The AUC only looks at how people are ranked. Asking also for a
Brier score close to the best one, so that the predicted risks are well
calibrated, leaves fewer models that put a person on different sides of 0.5.
All the boostings that meet each definition are used (`e3_sensitivity.csv`):

| Equally good means | Models | Prediction differs in at least one model | Between two models |
|---|---|---|---|
| AUC within 0.01 of the best | 144 | 25.1% | 4.0% |
| ... and Brier score within 0.005 of the best | 126 | 16.5% | 3.1% |
| ... and Brier score within 0.002 of the best | 83 | 10.9% | 2.2% |
| ... and Brier score within 0.001 of the best | 44 | 9.0% | 1.8% |

What the numbers show:

- For the data set as a whole, the 50 models tell the same story: the number of
  priors comes first and age second in every model, ahead of the five others.
- For one person, the story is less stable, and so is the decision: for 23% of
  people, at least one of these equally accurate models gives a different
  yes/no prediction from the others. Between two models drawn at random, the
  prediction differs for 4.1% of people, and most of the people concerned have
  a predicted risk close to 0.5.
- Readable models are not immune. Equally accurate logistic regressions also
  change the decision about some people: between two models, for 4.2% of
  people when all rows are used and 6.1% with random subsets of rows, against
  4.1% for the boostings. Over the 50 models, the regressions trained on all
  rows change it for 16% of people, against 23% for the boostings; among the
  people concerned, 93% have a risk between 0.4 and 0.6, against 80% for the
  boostings. The instability thus comes largely from the multiplicity of good
  models (the Rashomon effect), which readable models share.
- The size of these shares depends on the definition of "equally good": with a
  condition on the Brier score as well, the 23% falls to between 9% and 17%.
- An explanation, like the reading of one readable model, describes one model,
  not what another equally good model would decide.

One example from the test set: a 61-year-old man with 20 priors and a
non-felony charge. Model A (AUC 0.727) gives him a risk of 0.65 and its largest
SHAP value is the number of priors (+1.94 against -1.00 for age). Model B (AUC
0.721) gives him a risk of 0.34 and its largest SHAP value is age (-1.33 against
+1.18 for priors). He is the most extreme disagreement found by searching all
pairs of models and all people: he illustrates the effect, he does not measure
it.
