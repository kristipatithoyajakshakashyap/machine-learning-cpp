# Advanced EDA report

Titanic goal: understand survival predictors. Data dictionary: Pclass=ticket class; Age=years; Fare=ticket fare; SibSp/Parch=family counts; Sex=recorded category; Survived=binary target. PassengerId is excluded.

All Titanic analysis uses the seeded 80% training partition. Read quality.csv, age_missing_by_sex.csv, quantiles_outliers.csv, pairwise_association.csv and survival_by_sex.csv with the companion interpretations.

Recommended modeling actions: training-fitted age/fare imputation, standardized numeric predictors, explicit categorical encoding and majority baseline. Compare family-group validation before deployment. No test-set-derived preprocessing decisions. AirPassengers is a separate temporal example, not a Titanic feature.
