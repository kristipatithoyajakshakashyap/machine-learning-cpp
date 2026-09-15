#ifndef DSTS_TIMESERIES_HPP
#define DSTS_TIMESERIES_HPP

// Time-series helpers for module 09.
// pandas equivalents: pandas.Series.rolling(...).mean(),
// pandas.Series.autocorr(lag), numpy.diff.

#include <cstddef>
#include <vector>

#include "dsts/dtypes.hpp"
#include "dsts/series.hpp"

namespace dsts {

// Trailing mean over a window: result[row] is the mean of
// x[row-window+1 .. row]. The first window-1 cells are missing (NA), like
// pandas rolling(window).mean().
Series rolling_mean(const Series& x, size_t window);

// Pearson correlation between x[t] and x[t-lag] over the overlapping
// region; NaN when there is clean data for fewer than 2 pairs.
double autocorr(const Series& x, size_t lag);

// First differences x[t] - x[t-1]; position 0 is NA.
Series diff(const Series& x);

// Root mean squared error between two equal-length series (scipy.sk.metrics rmse).
double rmse(const Series& actual, const Series& predicted);

}  // namespace dsts

#endif  // DSTS_TIMESERIES_HPP