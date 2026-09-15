#ifndef DSTS_EDA_HPP
#define DSTS_EDA_HPP

// Exploratory-data-analysis helpers for module 03.
// The pandas equivalents: df.describe(), df.value_counts(), np.histogram,
// df.corr(), df.groupby(...).mean().

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "dsts/dataframe.hpp"
#include "dsts/dtypes.hpp"
#include "dsts/series.hpp"

namespace dsts {

// Frequency table for one string column, ordered by count descending
// (ties alphabetically), like pandas.Series.value_counts().
std::vector<std::pair<std::string, size_t>> value_counts(
    const DataFrame& df, const std::string& col);

// Multi-column block like pandas.DataFrame.describe(): each numeric column
// gets count/mean/std/min/25%/50%/75%/max on one line.
std::string summary(const DataFrame& df);

// Equal-width histogram from min to max. Missing cells are ignored.
// If every present value is identical the histogram has count == x.size()
// in the first bin.
struct Histogram {
  std::vector<double> edges;    // bins+1 values, ascending
  std::vector<size_t> counts;   // bins values
};
Histogram histogram(const Series& x, size_t bins);

// Pearson correlation matrix over the numeric columns of df.
std::vector<std::vector<double>> correlation_matrix(const DataFrame& df);

// Mean of one numeric column inside every group of a string column
// (pandas: df.groupby(group_col)[value_col].mean()).
struct GroupStat {
  std::string group;
  size_t count;
  double mean;
  double sd;  // sample sd within the group
};
std::vector<GroupStat> group_mean(const DataFrame& df,
                                  const std::string& group_col,
                                  const std::string& value_col);

}  // namespace dsts

#endif  // DSTS_EDA_HPP