#ifndef DSTS_JOIN_HPP
#define DSTS_JOIN_HPP

// Relational and reshaping helpers for module 05.
// The pandas/numpy equivalents:
//   concat_rows  ->  pandas.concat(axis=0)
//   merge        ->  pandas.merge(how="inner"|"left")
//   pivot_table  ->  pandas.DataFrame.pivot_table(aggfunc="mean")
//   melt         ->  pandas.DataFrame.melt()

#include <string>
#include <vector>

#include "dsts/dataframe.hpp"

namespace dsts {

// Row-bind several frames that share the exact same column list and types.
DataFrame concat_rows(const std::vector<DataFrame>& frames);

// Merge on one key column (present in both frames, same type). The key values
// of the right frame must be unique. The right columns except the key are
// appended; unmatched left-join cells become NA (numeric) / empty (string).
DataFrame merge(const DataFrame& left, const DataFrame& right,
                const std::string& key, const std::string& how = "inner");

// Mean of one numeric column inside every (index_col, columns_col) cell,
// laid out wide: first column = index_col (same type as the source), then one
// numeric column per distinct columns_col value; empty cells are NA.
DataFrame pivot_table(const DataFrame& df, const std::string& index_col,
                      const std::string& columns_col,
                      const std::string& value_col);

// Stack numeric columns into long form: id_cols repeat, plus a string
// "variable" column and a numeric "value" column (pandas df.melt()).
DataFrame melt(const DataFrame& df, const std::vector<std::string>& id_cols,
               const std::vector<std::string>& value_cols);

}  // namespace dsts

#endif  // DSTS_JOIN_HPP