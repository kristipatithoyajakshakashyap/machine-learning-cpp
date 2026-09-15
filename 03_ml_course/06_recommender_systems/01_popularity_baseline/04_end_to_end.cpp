#include "06_recommender_systems/04_recommender_evaluation/Workflow.hpp"
int main(int argc, char **argv) {
  return rec_course::workflow<rec_course::Popularity>(
      RUN_OUTPUT_DIR,
      std::string(COURSE_ROOT) + "/06_recommender_systems/data/u.data", {0},
      argc, argv);
}
