# Module 07: Statistics and hypothesis tests

## Purpose
Move from "the numbers are different" to "is the difference real?". All
helpers live in `include/dsts/stats.hpp`. Every p-value was cross-checked
against scipy on the same files (tips.csv, titanic.csv, iris.csv,
penguins.csv).

## Prerequisites

Module 03_eda and 00_math_foundations/03_probability, 04_statistics.

## Lessons

| File | Target | What it teaches | Python equivalent | What it writes |
|------|--------|-----------------|-------------------|----------------|
| `01_confidence_intervals.cpp` | `s07_01_ci` | 95% CI for a mean (per day). Data: tips.csv (real data, 244 restaurant tips) | `scipy.stats.t.interval` | prints only |
| `02_tests_smoker.cpp` | `s07_02_ttest` | Welch t-test: smoker No vs Yes. Data: tips.csv (real data, 244 restaurant tips) | `scipy.stats.ttest_ind(equal_var=False)` | prints only |
| `03_chi_square.cpp` | `s07_03_chisq` | Chi-square independence: Sex/Pclass vs Survived. Data: titanic.csv (real data, 891 passengers) | `scipy.stats.chi2_contingency` | prints only |
| `04_normality.cpp` | `s07_04_norm` | One-sample KS test vs a normal. Data: iris.csv, tips.csv, penguins.csv (real data) | `scipy.stats.kstest` | prints only |
| `05_bootstrap.cpp` | `s07_05_boot` | Seeded bootstrap CI for the mean. Data: tips.csv (real data, 244 restaurant tips) | `numpy.percentile` of resampled means | prints only |
| `06_anova.cpp` | `s07_06_anova` | One-way ANOVA + Kruskal-Wallis: tip by day. Data: tips.csv (real data, 244 restaurant tips), tip grouped by day | `scipy.stats.f_oneway`, `scipy.stats.kruskal` | prints only |
| `07_nonparametric.cpp` | `s07_07_nonparam` | Mann-Whitney U vs Welch t: Fare by Survived. Data: titanic.csv (891 passengers), Fare split by Survived | `scipy.stats.mannwhitneyu`, `ttest_ind` | prints only |
| `08_effect_sizes_multiple_testing.cpp` | `s07_08_effect` | Cohen's d, Cliff's delta, Bonferroni + Holm over six tip-by-day pairs, and a seeded t-test power curve (`corrections.csv`, `power_curve.svg`). Data: tips.csv (244 tips) - tip by day, all six day pairs | `pingouin.compute_effsize`, `multipletests`, `TTestIndPower` | `corrections.csv`, `power_curve.svg` under `results/08_effect_sizes_multiple_testing_results/` |
| `09_bayesian_basics.cpp` | `s07_09_bayes` | Beta-binomial survival by sex: posterior mean, 95% credible interval by grid vs Wald CI (`posterior_grid.csv`, `posterior.svg`). Data: titanic.csv - Survived by Sex (female 233/314, male 109/577) | `scipy.stats.beta(a+s, b+f)` | `posterior.svg`, `posterior_grid.csv` under `results/09_bayesian_basics_results/` |

## Build and run

Configure once from the repository root, build a lesson target, then run the
executable from `build/02_data_science/07_statistics_tests/`:

```powershell
cmake --preset course                                   # once, from the repo root
cmake --build --preset course --target s07_01_ci
.\build\02_data_science\07_statistics_tests\s07_01_ci.exe
```

Every lesson links the `dsts` static library, reads its dataset from
`02_data_science/data/` (injected as `DATA_DIR`) and writes any output files
to `results/<source stem>_results/` next to its source (injected as
`RUN_OUTPUT_DIR`, created on first run).

## dsts functions used

Types: `dsts::AnovaResult`, `dsts::ConfidenceInterval`, `dsts::DataFrame`, `dsts::KruskalResult`, `dsts::KsResult`, `dsts::MannWhitneyResult`, `dsts::OptD`, `dsts::Series`, `dsts::TTestResult`.

- `01_confidence_intervals.cpp`: `fmt`, `mean_ci95`, `read_csv`
- `02_tests_smoker.cpp`: `fmt`, `read_csv`, `ttest_ind`
- `03_chi_square.cpp`: `chi2_independence`, `fmt`, `read_csv`
- `04_normality.cpp`: `fmt`, `ks_normal`, `read_csv`
- `05_bootstrap.cpp`: `bootstrap_mean`, `fmt`, `mean_ci95`, `read_csv`
- `06_anova.cpp`: `anova_oneway`, `fmt`, `kruskal_wallis`, `read_csv`
- `07_nonparametric.cpp`: `fmt`, `mann_whitney_u`, `read_csv`, `ttest_ind`
- `08_effect_sizes_multiple_testing.cpp`: `fmt`, `read_csv`, `ttest_ind`, `write_csv`, `write_svg_line`
- `09_bayesian_basics.cpp`: `fmt`, `read_csv`, `write_csv`, `write_svg_scatter`

## Key ideas
- Confidence intervals grow with spread, shrink with n. Saturday's tip
  CI is wider than Sunday's because its std is bigger.
- A t-test needs a direction and a p-value. Both tips tests fail to
  reject (p > 0.05). Smokers tip about the same.
- Chi-square is for counts. Sex vs Survived has p ≈ 4e-59. Pclass p ≈
  5e-23. Neither pattern is luck.
- A KS reading. sepal_length looks Normal (p ≈ 0.18), petal_length does
  not (the column mixes three species).
- The bootstrap respects the skew. Its 95% CI for total_bill
  (`[18.689, 20.990]`) hugs the normal-approx CI while needing no
  distributional shape assumption.

- ANOVA compares k means at once. Tip by day gives F ~ 1.7 (p ~ 0.17),
  but Kruskal-Wallis on ranks gives p ~ 0.036. The skewed tail of tips hides
  a shift in the typical tip that the mean-based test does not see.
- Rank tests shrug off skew. Titanic fares have a few 500-pound
  outliers. Mann-Whitney and Welch both reject, but only the rank test
  would survive capping those fares.
- Effect size before p-value. Thur vs Sun tips give d ~ -0.39 and raw
  p ~ 0.024, but after Bonferroni/Holm over six pairs nothing survives. The
  power curve says a medium effect needs ~64 per group to reach 0.8.
- Bayes with a flat prior agrees with the Wald CI. Female survival
  posterior Beta(234, 82). Mean 0.74, credible interval [0.691, 0.787].
  P(female rate > male rate) = 1 on the grid.

## Key takeaways
- A confidence interval widens with spread and narrows with n.
- Match the test to the data: t-test for means, chi-square for counts, rank tests for skew.
- Report effect sizes and correct for multiple comparisons before quoting a p-value.

## Next module

08_train_test - hold-out splits, stratification, sampling bias, bootstrap, k-fold.