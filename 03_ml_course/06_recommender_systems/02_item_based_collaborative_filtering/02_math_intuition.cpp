#include "06_recommender_systems/04_recommender_evaluation/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(sim(i,j)=dot(centered_item_i,centered_item_j)/(norm_i*norm_j), multiplied by overlap/(overlap+10). Retain the K most similar positive neighbors globally per item. prediction=user_mean+sum(sim*centered_rating)/sum(sim) over those neighbors the user rated; fall back to smoothed item mean if none apply.)LESSON"
      << "\n";
  rec_course::Popularity m;
  m.fit({{1, 1, 1, 5, 1}, {2, 2, 2, 4, 2}});
  for (const auto &r : rec_course::recommend(m, 1))
    std::cout << "item " << r.first << " score " << r.second << "\n";
}
