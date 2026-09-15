#ifndef DSTS_CLEANING_HPP
#define DSTS_CLEANING_HPP

// Data-cleaning helpers used in module 02.
// The pandas/sklearn equivalents: df.dropna, df.duplicated/drop_duplicates,
// Tukey's IQR outlier rule, pandas.get_dummies, sklearn MinMaxScaler /
// StandardScaler.

#include <string>
#include <utility>
#include <vector>

#include "dsts/dataframe.hpp"
#include "dsts/series.hpp"

namespace dsts {

// Drop rows that have any missing cell. When cols is non-empty only the
// listed columns are checked.
DataFrame dropna(const DataFrame& df, const std::vector<std::string>& cols = {});

// Row i is "duplicated" when the whole row equals an EARLIER row (like
// pandas.DataFrame.duplicated()).
std::vector<bool> duplicated(const DataFrame& df);

// Keep the first occurrence of every unique row.
DataFrame drop_duplicates(const DataFrame& df);

// Tukey's IQR fences: [q1 - 1.5*iqr, q3 + 1.5*iqr].
std::pair<double, double> iqr_fences(const Series& x);

// True where the value lies OUTSIDE the IQR fences. Missing cells are false.
std::vector<bool> outlier_mask(const Series& x);

// Replace one string column with a 0/1 indicator column per category
// (first-appearance order), like pandas.get_dummies.
DataFrame one_hot(const DataFrame& df, const std::string& col);

// Scale into [0, 1] with min/max (sklearn MinMaxScaler). When min == max
// every present value maps to 0.0; missing stays missing.
Series minmax_scale(const Series& x);

// Standardize to mean 0 / sd 1 using the POPULATION standard deviation
// (sklearn StandardScaler default). Missing stays missing.
Series zscore_scale(const Series& x);

}  // namespace dsts

#endif  // DSTS_CLEANING_HPP