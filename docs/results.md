# Results

All numbers are produced by the scripts in `src/` with the seed set in
`src/common.py`. Intervals are 95% bootstrap intervals.

## E1: what does interpretability cost in accuracy?

COMPAS data, 6,172 people. Run time: about 2 minutes.

| Method | Kind | AUC | Accuracy |
|---|---|---|---|
| Sparse logistic regression | interpretable | 0.731 [0.718, 0.744] | 0.683 [0.672, 0.695] |
| Decision tree (depth at most 3) | interpretable | 0.708 [0.694, 0.721] | 0.670 [0.658, 0.682] |
| Gradient boosting | black box | 0.736 [0.723, 0.748] | 0.683 [0.671, 0.694] |
| COMPAS score | reference | 0.710 [0.698, 0.722] | 0.661 [0.649, 0.673] |
| Rudin (2019) rule list | reference | not comparable | 0.667 [0.655, 0.679] |
| Always predict "no re-arrest" | baseline | | 0.545 |

Gap between the gradient boosting and each other method (positive means the
black box is better):

| Gradient boosting minus | AUC gap | Accuracy gap |
|---|---|---|
| Sparse logistic regression | 0.005 [0.001, 0.009] | -0.001 [-0.007, 0.005] |
| Decision tree (depth at most 3) | 0.028 [0.022, 0.034] | 0.013 [0.005, 0.021] |
| COMPAS score | 0.026 [0.014, 0.037] | 0.022 [0.009, 0.034] |
| Rudin (2019) rule list | not comparable | 0.016 [0.006, 0.025] |

What the numbers show:

- The black box and the sparse logistic regression have the same accuracy. The
  AUC gap of 0.005 is distinguishable from zero but too small to matter in
  practice.
- The published three-condition rule list is as accurate as the COMPAS score on
  our data (0.667 against 0.661).
- The small decision tree is slightly behind, so an interpretable model is not
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

The setting selected for the gradient boosting is the simplest of its grid: 300
trees with two leaves each. Such a model adds up one effect per variable and
does not combine variables. The data therefore do not reward a more complex
model.

## E3: do equally accurate models give the same explanation?

COMPAS data, 1,852 people in the test set. Run time: under 1 minute.

Of the 200 models trained, 144 are within 0.01 AUC of the best one (0.728). The
50 models kept have a test AUC between 0.719 and 0.728.

| Level | Measure | Value |
|---|---|---|
| Whole data set | Median Kendall tau between the SHAP rankings of two models | 0.90 |
| Whole data set | Lowest Kendall tau between two models | 0.62 |
| Whole data set | Variables ranked first by at least one model | 1 (number of priors) |
| One person | People with the same top variable in all 50 models | 64% |
| One person | People with a variable whose effect clearly changes sign between models | 11% |
| Prediction | Median gap between the highest and lowest risk given to one person | 0.17 |
| Prediction | People whose yes/no prediction differs between models | 23% |

Number of models, out of 50, that give each importance rank to each variable:

| Variable | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Number of priors | 50 | | | | | | |
| Age | | 50 | | | | | |
| Felony charge | | | 44 | 5 | | | 1 |
| Male | | | 3 | 42 | 4 | 1 | |
| Other juvenile offenses | | | 3 | 3 | 44 | | |
| Juvenile misdemeanors | | | | | | 27 | 23 |
| Juvenile felonies | | | | | 2 | 22 | 26 |

What the numbers show:

- For the data set as a whole, the 50 models tell the same story: the number of
  priors comes first and age second in every model. The disagreement is limited
  to the less important variables.
- For one person, the story is less stable. For 36% of people, the 50 models do
  not agree on which variable matters most.
- The predictions themselves differ. For 23% of people, some of these equally
  accurate models predict re-arrest and others do not, and the risk given to a
  typical person varies by 0.17 from one model to another.

One example from the test set: a 61-year-old man with 20 priors and a
non-felony charge. Model A (AUC 0.727) gives him a risk of 0.65 and its largest
SHAP value is the number of priors (+1.94 against -1.00 for age). Model B (AUC
0.721) gives him a risk of 0.34 and its largest SHAP value is age (-1.33 against
+1.18 for priors). Both explanations are faithful to their own model, and the
two models are equally accurate, yet they reach opposite conclusions for this
person.
