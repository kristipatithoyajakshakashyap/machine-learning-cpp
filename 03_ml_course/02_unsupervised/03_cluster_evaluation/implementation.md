# Implementation walkthrough

The implementation uses shared silhouette_samples and silhouette. The project compares K-means and Ward across k=2..6, writes per-row scores and five 80% subsample ARI stability checks, and preserves the selected K-means state. This is internal model selection, not cross-validation. DBSCAN noise is removed from silhouette and its coverage is reported separately.

Workflow: load data ? inspect EDA ? fit preprocessing ? compare configurations ? assess stability or hold-out reconstruction ? save metrics and fitted state ? reload and compare outputs. Species labels never enter unsupervised fitting or parameter selection.
