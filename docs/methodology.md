# Methodology

This document describes the data, the models and the evaluation protocol of each
experiment, and the limits of what each experiment can show.

## What we call interpretable

A model counts as interpretable if a person can read the whole model: a linear
model with at most 10 variables, or a decision tree of depth at most 3 (at most
8 leaves). A black box is a model whose predictions cannot be read from its
structure, such as a sum of hundreds of trees.

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
| Sparse logistic regression (L1 penalty) | interpretable | yes |
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
   higher score than a person who was not re-arrested) and accuracy (share of
   correct yes/no predictions, with a threshold of 0.5 for the trained models
   and a score of 5 or more for COMPAS).
4. **Uncertainty.** We resample the 6,172 people with replacement 2,000 times
   and recompute every measure and every gap on each resample. The interval
   reported is the central 95% of these values. If the interval of a gap
   contains zero, the gap cannot be told apart from chance.

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
