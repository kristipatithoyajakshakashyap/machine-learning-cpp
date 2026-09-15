#include "06_recommender_systems/04_recommender_evaluation/Workflow.hpp"
#include <iostream>
int main() {
  std::cout
      << R"LESSON(Recall@10=hits/relevant_test_items. DCG@10=sum(1/log2(rank+1)) for binary-relevant recommendations; NDCG=DCG/ideal_DCG. If one of two relevant items appears at rank 1, recall=1/2 and NDCG=1/(1+1/log2(3)). Evaluate each user then average across eligible users.)LESSON"
      << "\n";
  rec_course::Popularity m;
  m.fit({{1, 1, 1, 5, 1}, {2, 2, 2, 4, 2}});
  for (const auto &r : rec_course::recommend(m, 1))
    std::cout << "item " << r.first << " score " << r.second << "\n";
}
