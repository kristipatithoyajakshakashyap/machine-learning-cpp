#ifndef DSTS_FEATURES_HPP
#define DSTS_FEATURES_HPP

// Feature-engineering helpers for module 06.
// pandas/numpy equivalents: pandas.cut, pandas.qcut, numpy.log1p,
// scipy.stats.skew.

#include <string>
#include <vector>

#include "dsts/series.hpp"

namespace dsts {

// Bin x into intervals defined by edges (edges.size() == labels.size() + 1).
// Matching pandas, the first interval is [edges[0], edges[1]] and the later
// ones are (edges[i], edges[i+1]] (right-closed, left-open). Values above
// edges.back() clamp to the last label; missing cells become "NA".
std::vector<std::string> cut(const Series& x, const std::vector<double>& edges,
                             const std::vector<std::string>& labels);

// Quantile edges that split x into q equal-count bins: q+1 edges produced
// by the linear-interpolation quantiles at k/q (k = 0..q).
std::vector<double> qcut_edges(const Series& x, size_t q);

// Sample skewness g1 (0 = symmetric, >0 right tail, <0 left tail).
double skew(const Series& x);

// log1p(v) = ln(1+v), stable near zero; propagates missing cells.
Series log1p(const Series& x);

}  // namespace dsts

#endif  // DSTS_FEATURES_HPP