# Data quality

Training rows: 712; full-row duplicates: 0. PassengerId is an identifier and is excluded from correlation/model features.

Check nonnegative fares, ages in [0,120], target in {0,1}, and Pclass in {1,2,3}. Missingness may be informative; do not silently remove all incomplete passengers.
