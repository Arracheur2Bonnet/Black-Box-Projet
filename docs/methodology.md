# Methodology

This document describes the data, the models and the evaluation protocol of each
experiment, and the limits of what each experiment can show.

## What we call interpretable

A model counts as interpretable if a person can read the whole model: a linear
model with at most 10 variables, or a decision tree of depth at most 3 (at most
8 leaves). A black box is a model whose predictions cannot be read from its
structure, such as a sum of hundreds of trees.

The additive models of E2b have no short formula and go beyond this criterion.
We report them, marked as such, because Rudin (2019) counts additive models as
interpretable: each of their curves can be plotted and read.

## E1: what does interpretability cost in accuracy?

Script: `src/e1_compas.py`

### Data

We use the COMPAS file published by ProPublica (Broward County, Florida) with
their standard filter: screening within 30 days of the arrest, known recidivism
status, ordinary charge degree and valid score. This leaves 6,172 people, of
whom 45.5% were re-arrested within two years. The outcome to predict is this
re-arrest.

The models receive seven inputs, all known at the time of the decision: age,
sex, number of prior offenses, three counts of juvenile offenses, and whether
the current charge is a felony.

- Race is not given to the models. The rule list of Rudin (2019) does not use
  it, and the fairness debate around COMPAS is a separate question from the one
  studied here. Sex is kept because that rule list uses it.
- The COMPAS score and every column recorded after the decision are never used
  as inputs, because they would leak the answer.

### Methods compared

| Method | Kind | Trained here |
|---|---|---|
| Sparse logistic regression (L1 penalty, intercept not penalised) | interpretable | yes |
| Decision tree of depth at most 3 | interpretable | yes |
| Gradient boosting (LightGBM) | black box | yes |
| COMPAS score (1 to 10) | reference | no |
| Rule list published in Rudin (2019), Figure 3 | reference | no |
| Always predict "no re-arrest" | baseline | no |

The rule list is applied exactly as published: predict re-arrest if the person
is male and aged 18 to 20, or aged 21 to 23 with 2 or 3 priors, or has more than
3 priors. Angelino et al. (2018) print the second age range as 21 to 22; we use
the 2019 version.

### Protocol

1. **Cross-validation.** The data are split in 10 parts. Each part is predicted
   by a model trained on the other 9, so every person is predicted by a model
   that never saw them.
2. **Same tuning effort for every model.** Each trained model has a grid of six
   candidate settings. The setting is chosen by a 5-fold search inside the
   training parts only. The grid of the gradient boosting includes very simple
   versions, because a large boosting model overfits on about 6,000 rows and
   would make the black box look worse than it can be.
3. **Measures.** AUC (the probability that a re-arrested person receives a
   higher score than a person who was not re-arrested), accuracy (share of
   correct yes/no predictions, with a threshold of 0.5 for the trained models
   and a score of 5 or more for COMPAS) and, for the trained models only, the
   Brier score (mean squared gap between the predicted probability and the
   outcome; lower is better). The COMPAS score and the rule list give no
   probability, so the Brier score does not apply to them. The setting chosen
   in each of the 10 folds is recorded in `outputs/e1_models.txt`.
4. **Uncertainty.** We resample the 6,172 people with replacement 2,000 times
   and recompute every measure and every gap on each resample. The interval
   reported is the central 95% of these values. If the interval of a gap
   contains zero, the gap cannot be told apart from chance.
5. **Same people, different answers (added after review).** An average gap
   can hide individual differences, so we also count the people whose yes/no
   prediction differs between the black box and each other method, and we
   check whether their predicted risk is close to the threshold of 0.5.

### Limits

- With seven simple inputs, a black box has little room to do better. A small
  gap here is expected and does not show that black boxes never help.
- All seven inputs are available to every model, so the "at most 10 variables"
  criterion does not separate the models here. What differs is the form of the
  model.
- The comparison with the COMPAS score is not on equal terms. Our models are
  trained on these Broward County data, whereas COMPAS was built elsewhere, for
  a broader purpose and from other inputs. The result says that simple models
  do at least as well on these data, not that COMPAS is a poor tool.
- The rule list gives a yes/no answer and not a score, so its AUC is not
  comparable with the others. Only its accuracy should be compared.
- The accuracy figures of Angelino et al. (2018), 0.665 for the rule list and
  0.660 for COMPAS, were obtained on 6,907 people with a different filter. They
  are close to ours but are not the same measurement.
- "Recidivism" in these data means a new arrest, not a proven offense (Bao et
  al., 2021), and Barenstein (2019) reports a processing error in the two-year
  window of the ProPublica file.
- The intervals reflect the uncertainty from the people sampled, not the
  variation between two trainings of the same model.

## E2a: does the black box win at forecasting unemployment?

Script: `src/e2a_unemployment.py`

### Task and data

We forecast the change in the US unemployment rate over the next 12 months.
The task follows the Bank of England study of Buckmann and Joseph (2023, first
published as a working paper in 2022), who report lower errors for machine
learning models than for linear models on it. In their results the advantage
over a simple autoregressive model is not statistically significant.

The data are the FRED-MD panel of the Federal Reserve Bank of St. Louis
(McCracken and Ng, 2016): 126 monthly US macroeconomic series (output, labour
market, housing, money, interest rates, prices, stock market). We use the
release of September 2026 and the months from January 1960 to September 2025,
the last month before a gap in the unemployment series. Each series is made
stable over time with the transformation recommended by the authors of the
panel (for example a growth rate instead of a level). The 119 series without
missing values are kept.

The inputs known at the end of a month are the average of the last three
monthly values of each of the 119 series, plus the current unemployment rate:
120 inputs in total.

### Methods compared

| Method | Kind | Inputs |
|---|---|---|
| Naive: unemployment does not change | baseline | none |
| Autoregressive model | interpretable | last 6 unemployment rates |
| Sparse linear model (Lasso), at most 10 series | interpretable | the 120 inputs |
| Gradient boosting (LightGBM) | black box | the 120 inputs |

The sparse linear model and the black box have access to the same panel. The
difference is that the linear model may use at most 10 series, which is our
criterion for an interpretable model.

### Protocol

1. **Rolling origin.** A forecast for month T is made at the end of month T-12.
   The model that makes it was fitted only on examples whose 12-month outcome
   was already known at that date. Random cross-validation is never used on
   time series, because it would let a model learn from the future.
2. **Yearly refit.** The models are re-estimated once a year on all the data
   available at that date. Forecasts cover January 1990 to September 2025 (429
   months).
3. **Same tuning effort.** The sparse linear model chooses among six sizes (1,
   2, 3, 5, 8 or 10 series) and the gradient boosting among six settings. Both
   choices are made on the last ten years of the training window, which the
   candidates have not seen.
4. **Measure.** The root mean squared error (RMSE) in percentage points of
   unemployment, over all months and without the Covid period (forecasts for
   March 2020 to December 2021). The jump of April 2020 could not be forecast
   by any model and dominates the error when it is included.
5. **Uncertainty.** Forecast errors of neighbouring months are related, because
   12-month windows overlap. We therefore resample the months in blocks of 24
   consecutive months, 2,000 times, and report the central 95% of the values.

### Limits

- The intervals are wide. There are only a few recessions in 35 years, and they
  account for most of the error, so differences between methods are hard to
  establish. The only gap whose interval excludes zero (autoregression against
  the naive forecast) was added after review, among twelve intervals computed
  without any correction for multiple comparisons.
- The Covid period was left out in the protocol, before the forecasts were
  run, but it holds the months in which the autoregression fails most. With
  them, the autoregression has the largest error instead of the smallest, so
  the ranking of the models is not robust; only the absence of a significant advantage for the black box holds in both cases.
- We use one release of the data. Many series are revised after their first
  publication and are published with a delay, so a forecaster at the time had
  slightly different and older numbers.
- Our result differs from the one of Buckmann and Joseph (2023). Their study
  uses a different selection and preparation of the inputs, other models and a
  different evaluation period. Our experiment is not a replication of theirs
  and does not show that their result is wrong.
- Only one black box and one way of preparing the inputs were tried. We did not
  search for a setting in which the black box wins, because choosing the
  protocol after seeing the test errors would bias the comparison.

## E2b: does the black box win at forecasting electricity demand?

Script: `src/e2b_electricity.py`

### Why this problem

E1 and E2a showed little or no advantage for the black box. We therefore added a
problem on which a black box is expected to do well, and we wrote this
expectation down before running the experiment. Electricity demand depends on
temperature in a non-linear way: it rises when it is cold (heating) and when it
is hot (air conditioning). This effect is not the same at every time of day or
on every type of day, and the data contain tens of thousands of observations. A
linear model with ten variables cannot represent such combined effects, whereas
a flexible model can. The result of E2a is kept whatever E2b shows.

### Task and data

The data are the half-hourly electricity demand of the state of Victoria
(Australia) from 2012 to 2014, with the temperature in Melbourne and the public
holidays: 52,608 half-hours. This is the `vic_elec` data set used in the
textbook of Hyndman and Athanasopoulos, *Forecasting: Principles and Practice*.

Once a day, each model forecasts the demand of each of the next 48 half-hours.
All models receive the same information:

- the time of day, the day of the week, whether the day is a working day
  (neither a weekend nor a public holiday) and whether the day before was one;
- the temperature at the time being forecast and at the same time the day
  before;
- the demand at the same time the day before and one week before, and the
  average demand of the day before.

### Methods compared

| Method | Kind | What can be read |
|---|---|---|
| Naive: demand of the same half-hour one week earlier | baseline | |
| Linear regression with 9 variables | interpretable | one coefficient per variable |
| Additive model | readable curve by curve | one curve per input |
| Linear regression per half-hour (added after review) | interpretable | 48 equations of 9 coefficients |
| Additive model per half-hour (added after review) | readable map by map | one map (half-hour x input) per input |
| Gradient boosting (LightGBM) | black box | nothing directly |

- **Linear regression.** A straight line cannot bend, so temperature enters
  through heating degrees and cooling degrees (how far the temperature is below
  or above 18 °C), the usual hand-made transformation. The time of day is not
  given as such; it is carried by the past demand at the same half-hour.
- **Additive model.** It is a gradient boosting in which every tree has only two
  leaves and therefore looks at one input. The model is then a sum of nine
  curves, one per input, and each curve can be plotted and read. It can bend,
  but it cannot represent an effect that depends on two inputs at once, for
  example a temperature effect that differs between 3 a.m. and 6 p.m. This
  model has no short formula, so it goes beyond our strict criterion. We
  include it for two reasons. Rudin (2019) describes additive models as "a
  known form of interpretable model", and additive models are an established
  tool for electricity load forecasting (Fan and Hyndman, 2012). A linear
  regression alone would understate what a readable model can do.
- **Linear regression per half-hour.** The same 9 variables, but one equation
  for each of the 48 half-hours of the day, so that every effect (heat, cold,
  working day, past demand) may change with the time of day. Each equation
  respects our limit of 10 variables, but the model has 480 numbers in all (48
  equations of an intercept and 9 coefficients). They are meant to be read as
  curves over the day, one per variable, rather than one by one. This is the
  structure of the load forecasting models of Fan and Hyndman (2012), who fit a
  separate model for each half-hour.
- **Additive model per half-hour.** A gradient boosting whose trees have three
  leaves and may only use the half-hour and one other input. The model is a sum
  of eight maps, one per input, each showing how the effect of that input
  changes with the time of day. Like the additive model, it has no short
  formula and goes beyond our criterion.
- **Gradient boosting.** Its trees have several leaves and can combine inputs.

The two models per half-hour were added after a review of the first version of
this experiment, which compared the black box only with readable models that
cannot let an effect change with the time of day. They were specified from the
literature before being run, and their settings were chosen like the others.

### Protocol

1. **Test year.** The models are evaluated on the whole year 2014 (365 days,
   17,520 half-hours). They are refitted at the start of each month on all the
   earlier data.
2. **Settings chosen before the test year.** The number of trees of the two
   additive models (six candidates each) and the size and number of trees of
   the gradient boosting (six candidates) are chosen once, on the last eight
   weeks of 2013. The linear regressions have no setting.
3. **Measures.** The mean absolute percentage error (MAPE), which is the average
   error as a percentage of the actual demand, and the RMSE.
4. **Uncertainty.** We resample the 365 test days in blocks of 7 consecutive
   days, 2,000 times, and report the central 95% of the values.
5. **Peaks.** We also report the errors on the 1% of half-hours with the highest
   demand (175 half-hours from 11 days, mostly one heat wave). This table was
   added after looking at the forecasts of the hottest week and has no
   interval, so it is a description and not a planned test.

### Limits

- The temperature of the forecast day is taken as known, as if the weather
  forecast were perfect. All models receive it, but real errors would be larger,
  and a model that relies more on temperature would suffer more.
- The data cover one region and three years, and the test covers one year.
- The problem was added after E1 and E2a had shown little or no advantage for the black box, and the two models per half-hour were added after a review. Both
  additions follow the literature, but other readable models could do better
  or worse.
- The problem was chosen because the black box was expected to win. The result
  shows that the size of the cost of interpretability depends on the structure
  given to the readable model. It does not say how common such problems are.
- A demand forecast is not a decision about a person. The stakes that motivate
  the argument of Rudin (justice, credit, health care) are of another kind.

## E3: do equally accurate models give the same explanation?

Script: `src/e3_rashomon.py`

### Idea

For one data set there are usually many different models with almost the same
accuracy. This is called the Rashomon effect. If a black box is explained after
the fact, the explanation describes one of these models. E3 asks whether another
model, just as accurate, would have given the same explanation.

### Protocol

1. The COMPAS data of E1 are split once: 70% to train, 30% (1,852 people) to
   test.
2. We train 200 gradient boosting models on the same training data. Each has
   settings drawn at random (size and number of trees, learning rate, share of
   rows and of variables used by each tree) and its own random seed.
3. A model is "nearly as good as the best" if its test AUC is within 0.01 of the
   best one. We draw 50 of these models at random.
4. Each of the 50 models is explained with SHAP. For one person, the SHAP value
   of a variable says how much that variable pushed the prediction of this model
   above or below its average prediction.
5. We compare the 50 explanations at two levels.
   - **Whole data set.** Variables are ranked by their average absolute SHAP
     value. Two rankings are compared with Kendall's tau, which goes from 1
     (same order) to -1 (reversed order). A variable that a model does not use
     at all has an importance of zero and is ranked last.
   - **One person.** We look at the variable with the largest absolute SHAP value for
     that person, and at whether a variable clearly raises the risk in one model
     and clearly lowers it in another (effects below 0.1 in absolute value are
     ignored).
6. We also compare the predictions themselves: the gap between the highest and
   the lowest risk given to the same person by the 50 models, and whether the
   yes/no prediction changes. A person is counted as soon as one of the 50
   models disagrees with the others, so this share grows with the number of
   models compared. We therefore also give the share of people whose
   prediction differs between two models, averaged over all pairs (added after
   review).
7. **Control (added after review).** We repeat steps 2 to 6 with 200 sparse
   logistic regressions, each with its own penalty, in two versions. In the
   first, each regression is trained on a random 50 to 90% of the training
   rows, the share that each tree of a boosting sees. In the second, every
   regression sees all the rows and only the penalty varies. For a linear
   model, the contribution of a variable to one prediction is its coefficient
   times the gap between the person's value and the average value, which is
   also its exact SHAP value. If equally accurate readable models also
   disagree, part of the disagreement comes from the multiplicity of good
   models (the Rashomon effect), not from post-hoc explanations.
8. **Uncertainty.** The shares of people are given with a 95% interval,
   obtained by resampling the 1,852 test people 2,000 times.

The example of one person shown in the figure is the most extreme disagreement
found by searching all pairs of models and all people. It illustrates the
effect; it does not measure it.

### Limits

- E3 does not test the fidelity of an explanation to its model: for a sum of
  trees, SHAP values are computed exactly. It tests what the explanation of one
  model says about other models that are just as accurate.
- The models are selected on the same test set that is used to compare them.
  This is acceptable here because we do not report their accuracy as a result;
  we only need them to be about equally accurate.
- With seven inputs, of which two dominate, there is little room for rankings to
  differ. A data set with many correlated inputs would be a harder test.
- The share of 23% grows with the number of models compared; the share between
  two models (4.1%) does not. It also depends on what "equally good" means: with
  a condition on the Brier score as well as on the AUC, it falls from 25% (all the boostings within 0.01 of AUC) to between 9% and 17% (`outputs/e3_sensitivity.csv`). The design of the control matters as
  well: the logistic regressions disagree for 25% of people when each sees part
  of the rows, and for 16% when all see every row.
- SHAP describes what a model does, not what causes re-arrest.

## References

- Angelino, E., Larus-Stone, N., Alabi, D., Seltzer, M. and Rudin, C. (2018).
  Learning certifiably optimal rule lists for categorical data. *Journal of
  Machine Learning Research* 18(234), 1–78.
- Bao, M. et al. (2021). It's COMPASlicated: the messy relationship between RAI
  datasets and algorithmic fairness benchmarks. *NeurIPS Datasets and
  Benchmarks*.
- Barenstein, M. (2019). ProPublica's COMPAS data revisited. arXiv:1906.04711.
- Buckmann, M. and Joseph, A. (2023). An interpretable machine learning workflow
  with an application to economic forecasting. *International Journal of
  Central Banking* 19(4), 449–522. First published as Bank of England Staff
  Working Paper No. 984 (2022).
- Fan, S. and Hyndman, R. J. (2012). Short-term load forecasting based on a
  semi-parametric additive model. *IEEE Transactions on Power Systems* 27(1),
  134–141.
- Hyndman, R. J. and Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice*, 3rd edition. OTexts.
- Lundberg, S. and Lee, S.-I. (2017). A unified approach to interpreting model
  predictions. *Advances in Neural Information Processing Systems* 30.
- McCracken, M. W. and Ng, S. (2016). FRED-MD: a monthly database for
  macroeconomic research. *Journal of Business & Economic Statistics* 34(4),
  574–589.
- Rudin, C. (2019). Stop explaining black box machine learning models for high
  stakes decisions and use interpretable models instead. *Nature Machine
  Intelligence* 1, 206–215.
