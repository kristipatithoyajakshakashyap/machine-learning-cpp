# Theory

## Why combine models

A single model has one inductive bias. Logistic regression draws linear boundaries, kNN follows local density, a forest carves axis-aligned boxes. Each makes systematic mistakes in the regions its bias handles badly. If those regions differ between models, a combination can be right where any one member is wrong. That is the whole promise of ensembling and it rests on two conditions: the members must be better than chance and their errors must not be perfectly correlated.

## Voting

Hard voting treats each member as one vote for its predicted label and returns the majority. It ignores confidence: a member that is 51% sure counts as much as one that is 99% sure. With three members and three classes the vote can split three ways; this implementation resolves ties by the smallest class index, which is arbitrary but deterministic.

Soft voting averages the members' probability rows and takes the argmax. It uses confidence, so a very sure member can outvote two hesitant ones. It requires every member to output probabilities on a comparable scale. kNN class fractions in steps of 1/5 and forest leaf averages are coarser than logistic softmax outputs, so soft voting quietly weights the smoother member more where the others are uncertain.

## Stacking

Voting fixes the combination rule in advance. Stacking learns it: a second-level model (the meta-learner) takes the members' probabilities as its input features and is trained to predict the label. With a logistic meta-learner the result is a learned, per-class weighting of every member's every probability, including cross-terms such as "when kNN says class 1, discount the forest's class 2".

The trap is where the meta-learner's training data comes from. If the members are fitted on the training rows and then asked for probabilities on the same rows, those probabilities are optimistic: kNN with k = 5 is nearly always right on its own training points, and a forest interpolates its bootstrap samples. The meta-learner would learn to trust the members far more than they deserve and the ensemble would be worse than its best member on new data.

Out-of-fold (OOF) prediction removes the leak. Split the training rows into K folds; for each fold, fit the members on the other K - 1 folds and record their probabilities on the held-out fold. Every training row is scored by members that never saw it, so the meta-features have the same optimism as they will at test time. After the meta-learner is fitted, the members are refitted on all training rows for use at prediction time. This module uses K = 5 stratified folds with a fixed seed so that the OOF matrix, and therefore the meta-weights, are reproducible.

## When ensembles help, and when they do not

- Members with similar accuracy and different error patterns: gains are largest. Heterogeneous model families are the easiest way to get that diversity.
- One member dominates: averaging drags it toward the weaker members. Stacking can learn to down-weight the weak ones; voting cannot.
- Members that are nearly identical (three forests with different seeds): almost no gain beyond what more trees in one forest would give.
- Small datasets: the meta-learner adds parameters and the OOF matrix is noisy. Keep it simple (logistic, strong regularisation) and expect small, noisy differences between modes, as on wine.

Ensembles trade interpretability and inference cost for accuracy. The saved model here holds three full members plus the meta-weights; inference runs all three.
