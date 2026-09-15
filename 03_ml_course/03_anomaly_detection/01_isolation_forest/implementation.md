# Isolation Forest

IsolationForest.hpp samples without replacement, chooses a nonconstant feature and uniform threshold at each node, and stops at ceil(log2(sample_size)). Stored node indices make recursion and reload explicit. score_samples returns larger values for more unusual rows. The end-to-end project uses 60/20/20 train/calibration/test partitions and training-only preprocessing.

Training scans feature ranges recursively on each subsample, approximately O(trees * p * sample_size * log(sample_size)). Scoring traverses one path per tree. The validation-calibrated threshold is stored with the forest; fresh-process CLI inference reproduces both scores and flags.
