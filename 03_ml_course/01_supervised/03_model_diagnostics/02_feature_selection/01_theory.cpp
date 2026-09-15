#include "Model.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Feature selection removes inputs that do not help. Filters score each column against the target with no model (ANOVA F, correlation). Wrappers ask a model directly by adding features one at a time and cross-validating. Embedded methods let the fitting procedure shrink useless coefficients to zero (lasso). Selection must be fitted on training rows only; SelectedForest does that inside fit() so the pipeline's cross-validation stays honest.)LESSON"
      << "\n";
  ml::Mat X = {{0, 5, 1}, {0.2, 3, 1}, {0.1, 4, 1}, {3, 5, 1}, {3.1, 3, 1}, {2.9, 4, 1}};
  ml::Vec y = {0, 0, 0, 1, 1, 1};
  ml::seed_rng(1);
  course::SelectedForest model(1);
  model.fit(X, y);
  std::cout << "kept feature: " << model.kept()[0] << " (scores "
            << ml::print_vec(model.scores()) << ")\n";
}
