#include "Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Hyper-parameters are choices the learner cannot fit from data (tree depth, leaf size, regularisation strength). They must be chosen by an outer loop that scores each candidate on rows the fit never saw. Grid search enumerates every combination; random search samples a subset and usually finds an equally good point far cheaper when only a few parameters matter. Whatever selects the parameters also consumes the validation rows: the best CV score is an optimistic estimate, and only nested cross-validation (or a separate hold-out) reports an honest number.)LESSON"
      << "\n";
  std::cout << "TunedForest grid: " << course::TunedForest::grid_size()
            << " combinations of max_depth x min_leaf\n";
  for (size_t i = 0; i < course::TunedForest::grid_size(); ++i)
    std::cout << "  index " << i << " -> max_depth "
              << course::TunedForest::depth_of(i) << ", min_leaf "
              << course::TunedForest::leaf_of(i) << "\n";
}
