#include "06_recommender_systems/04_recommender_evaluation/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(rating_hat=mu+b_u+b_i+p_u dot q_i. For residual e=r-rating_hat, SGD updates b_u += eta*(e-lambda*b_u), p_u += eta*(e*q_i-lambda*p_u), with the old p_u used for the paired q_i update. The implementation penalizes per observed rating and shuffles only the training interactions.)LESSON"
      << "\n";
  rec_course::Popularity m;
  m.fit({{1, 1, 1, 5, 1}, {2, 2, 2, 4, 2}});
  for (const auto &r : rec_course::recommend(m, 1))
    std::cout << "item " << r.first << " score " << r.second << "\n";
}
