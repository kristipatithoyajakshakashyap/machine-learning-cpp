# Theory: why and how to select features

A model that sees thirty columns when three carry the signal pays three prices: variance (every useless column is another way to fit noise), cost (training and inference scale with width) and interpretability (a thirty-term explanation is no explanation). Feature selection keeps a subset of the original columns; unlike PCA it does not invent new ones, so the kept features keep their names.

## Three families

**Filters** score each column against the target without fitting any model. For a class label the score is the one-way ANOVA F statistic (how far the class means are apart relative to the spread within each class); for a continuous target it is |Pearson r|. Filters are cheap, O(n p), and model-agnostic, but they look at one column at a time: a column that is useless alone but decisive together with another scores low, and two copies of the same column both score high.

**Wrappers** ask the model. Forward selection starts with no features, tries adding every remaining column, keeps the one that most improves the cross-validated metric, and repeats. It sees interactions and redundancy through the model's eyes, at the price of O(p k) model fits, each of them a full cross-validation.

**Embedded** methods fold selection into fitting. The lasso penalty lambda * sum |w_j| drives coefficients exactly to zero; sweeping lambda from large to small gives a path along which features "enter" one at a time. The lambda at which a feature enters is its rank. This costs one fit per lambda and handles redundancy (of two collinear columns the lasso keeps one), but it is a linear model's opinion.

## The leakage rule

Selection is part of training. If the columns are chosen by looking at all rows and the model is then cross-validated, every fold has already seen its test rows through the selection step and the CV score is optimistic. The only honest layout is: reserve the hold-out, and inside each fold select on the training rows only. `SelectedForest::fit` computes the filter scores on the rows it is given, which is exactly what makes it safe to hand to `run_supervised`.

## What `k` means

With a filter, the number of kept columns `k` is a hyperparameter like any other and is chosen by the pipeline's cross-validation over `{5, 10, 20, 30}`. A curve that flattens early says the extra columns add nothing; a curve that keeps rising says the filter is discarding columns the forest could use.
