// Include the actual taught implementation rather than duplicating its logic.
#define main normalizer_lesson_main
#include "../11_capstone/03_dataset_normalizer.cpp"
#undef main
#include <limits>
int main(){
  if(!min_max_normalize({}).empty())return 1;
  if(min_max_normalize({5,5,5})!=std::vector<double>({0,0,0}))return 1;
  if(min_max_normalize({-2,0,2})!=std::vector<double>({0,.5,1}))return 1;
  bool rejected=false;try{min_max_normalize({std::numeric_limits<double>::infinity()});}
  catch(const std::invalid_argument&){rejected=true;}
  return rejected?0:1;
}
