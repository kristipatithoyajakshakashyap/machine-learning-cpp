# Implementation walkthrough

GaussianMixture.hpp computes log densities and normalizes with log-sum-exp to avoid underflow. Variance flooring and empty-component recovery keep EM defined. The project compares component counts and three seeded initializations using BIC, then reports hard-assignment silhouettes and profiles. Saved means, variances and weights support probability prediction on new rows.

Workflow: load data ? inspect EDA ? fit preprocessing ? compare configurations ? assess stability or hold-out reconstruction ? save metrics and fitted state ? reload and compare outputs. Species labels never enter unsupervised fitting or parameter selection.

For n rows, p features and k components, each EM iteration costs O(nkp); stored responsibilities use O(nk) memory. Three initializations are compared, because EM reaches a local stationary point. Diagonal covariance saves parameters but cannot represent rotated within-component correlation directly.
