# Implementation walkthrough

DBSCAN.hpp precomputes all radius neighborhoods then expands core components with a queue. Labels -2 and -1 represent unvisited and noise. Noise encountered later from a core point becomes border. Complexity is O(n^2 p), appropriate for these teaching datasets. Snapshot stores fitted rows, core flags and assignments. No native predict is exposed.

Workflow: load data ? inspect EDA ? fit preprocessing ? compare configurations ? assess stability or hold-out reconstruction ? save metrics and fitted state ? reload and compare outputs. Species labels never enter unsupervised fitting or parameter selection.
