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
