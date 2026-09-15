// Uses the same saved pipeline as the training entry point; requires inference arguments.
#define main course_workflow_entry
#include "04_end_to_end.cpp"
#undef main
int main(int argc,char** argv) {
  if(!ml::inference_requested(argc,argv)) { std::cerr << "Use --predict input.csv --model run/model\n"; return 1; }
  return course_workflow_entry(argc,argv);
}
