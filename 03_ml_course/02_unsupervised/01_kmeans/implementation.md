# Implementation walkthrough

KMeans.cpp implements seeding, assignment, averaging, restart selection, and nearest-centroid prediction. Empty clusters keep their previous center. The final assignment is recomputed after convergence. fit validates dense finite inputs and positive budgets. Snapshots retain learned centers and labels.

Workflow: load data ? inspect EDA ? fit preprocessing ? compare configurations ? assess stability or hold-out reconstruction ? save metrics and fitted state ? reload and compare outputs. Species labels never enter unsupervised fitting or parameter selection.
