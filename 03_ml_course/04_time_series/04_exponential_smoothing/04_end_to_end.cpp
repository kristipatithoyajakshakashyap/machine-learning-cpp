#include "04_time_series/03_rolling_origin_evaluation/Workflow.hpp"
int main(int argc, char **argv) {
  return forecast_course::workflow(
      RUN_OUTPUT_DIR, std::string(DATA_DIR) + "/air_passengers.csv",
      forecast_course::Forecaster::HoltWinters, argc, argv);
}
