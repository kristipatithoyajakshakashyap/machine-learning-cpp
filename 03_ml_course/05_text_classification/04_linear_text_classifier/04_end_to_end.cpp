#include "05_text_classification/Workflow.hpp"
int main(int argc, char **argv) {
  return text_course::workflow<text_course::LinearTextClassifier>(
      RUN_OUTPUT_DIR,
      std::string(COURSE_ROOT) +
          "/05_text_classification/data/SMSSpamCollection",
      true, {0.0001, 0.001}, argc, argv);
}
