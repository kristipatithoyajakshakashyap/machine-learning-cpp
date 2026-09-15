// 13_numerical_computing/01_precision.cpp
// Floating-point pitfalls: catastrophic cancellation and equality tests.
//
// WHAT YOU LEARN:
//   - Naive summation of {1e16, 1, -1e16} returns 0: the 1 is lost when it
//     is added to 1e16 because a double has only ~16 significant digits.
//   - Neumaier's compensated summation keeps a running correction term and
//     recovers the exact answer 1.
//   - 0.1 + 0.2 != 0.3 in binary floating point; compare with a mixed
//     absolute/relative tolerance instead of operator==.
//
// Reads:   nothing (data is hard-coded).
// Writes:  results/01_precision_results/precision.csv (method,sum table with
//          17 significant digits so both results round-trip exactly).
// Run:     cmake --build --preset course --target c13_precision
//          then build/01_get_to_know/13_numerical_computing/c13_precision
//          Prints nothing; exit code 0 means both checks passed (used by CTest).

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <vector>

// Neumaier compensated sum of x.
//   x: values to add.
//   returns: sum + accumulated rounding correction.
// Each step computes t = sum + v and adds the rounding error of that
// addition to `correction`. Which operand is larger decides how the lost
// low-order bits are recovered.
double compensated_sum(const std::vector<double>& x){
  double sum=0,correction=0;
  for(double v:x){double t=sum+v;if(std::abs(sum)>=std::abs(v))correction+=(sum-t)+v;else correction+=(v-t)+sum;sum=t;}
  return sum+correction;
}

// Compare naive and compensated sums, write them to CSV, and self-test.
int main(){
  const std::vector<double> x{1e16,1,-1e16};
  const double naive=std::accumulate(x.begin(),x.end(),0.0),stable=compensated_sum(x);
  // RUN_OUTPUT_DIR is injected by CMake; create it lazily on first run.
  std::filesystem::create_directories(RUN_OUTPUT_DIR);
  // failbit|badbit exceptions turn a silent write failure into a thrown error.
  std::ofstream out;out.exceptions(std::ios::failbit|std::ios::badbit);out.open(RUN_OUTPUT_DIR "/precision.csv");
  out<<std::setprecision(17)<<"method,sum\nnaive,"<<naive<<"\nNeumaier,"<<stable<<"\n";
  // Tolerance = absolute floor (1e-12) + relative part scaled by the larger operand.
  double a=0.1+0.2,b=0.3;const bool close=std::abs(a-b)<=1e-12+1e-10*std::max(std::abs(a),std::abs(b));
  // Exit 0 only if the compensated sum is exactly 1 and the tolerance test passes.
  return stable==1&&close?0:1;
}
