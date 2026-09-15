# Implementation walkthrough

Agglomerative.cpp builds an n-1 row linkage matrix with leaf IDs 0..n-1 and merge IDs n+t. A dense distance table uses Lance-Williams updates; nearest-pair search costs O(n^3) overall and O(n^2) memory. cut(k) replays the first n-k merges. Saved states support cut and assignments, not arbitrary new-row prediction.

Workflow: load data ? inspect EDA ? fit preprocessing ? compare configurations ? assess stability or hold-out reconstruction ? save metrics and fitted state ? reload and compare outputs. Species labels never enter unsupervised fitting or parameter selection.
