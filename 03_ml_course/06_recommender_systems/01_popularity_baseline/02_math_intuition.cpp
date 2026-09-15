#include "06_recommender_systems/04_recommender_evaluation/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(rating_hat_i=(sum_ratings_i+10*global_mean)/(count_i+10). Ranking score is count_i. For count=2,sum=10,global_mean=3, the smoothed prediction is 40/12=3.333. Exclude items the user already rated.)LESSON"
      << "\n";
  rec_course::Popularity m;
  m.fit({{1, 1, 1, 5, 1}, {2, 2, 2, 4, 2}});
  for (const auto &r : rec_course::recommend(m, 1))
    std::cout << "item " << r.first << " score " << r.second << "\n";
}
