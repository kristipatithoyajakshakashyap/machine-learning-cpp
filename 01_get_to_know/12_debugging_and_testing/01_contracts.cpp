// 12_debugging_and_testing/01_contracts.cpp
// Preconditions as exceptions, checked by a tiny self-test harness.
//
// WHAT YOU LEARN:
//   - A function "contract" says what the caller must guarantee (here: at
//     least one finite observation). Violations throw, they do not return
//     garbage.
//   - A running-mean update (Welford style) avoids a large intermediate sum.
//   - A minimal check() lambda turns main() into a self-test: the process
//     exit code is non-zero when any check fails, so CTest can run it.
//   - Run under a debugger: break inside checked_mean and inspect x, mean, n.
//
// Reads:   nothing (data is hard-coded).
// Writes:  nothing (prints PASS/FAIL lines to stdout).
// Run:     cmake --build --preset course --target c12_contracts
//          then build/01_get_to_know/12_debugging_and_testing/c12_contracts
//
// EXPECTED OUTPUT:
//   PASS known mean
//   PASS singleton
//   PASS empty input rejected
//   PASS translation invariant

// Run under a debugger. Break inside checked_mean and inspect arguments.
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

// Arithmetic mean of x with explicit preconditions.
//   x: observations; must be non-empty and every value must be finite.
//   returns: the mean, computed with an incremental update so no partial sum
//            can overflow before division.
//   throws:  std::invalid_argument when a precondition is violated.
double checked_mean(const std::vector<double>& x){
  if(x.empty())throw std::invalid_argument("mean requires observations");
  double mean=0;size_t n=0;
  // mean_n = mean_{n-1} + (v - mean_{n-1}) / n : one pass, no separate sum.
  for(double v:x){if(!std::isfinite(v))throw std::invalid_argument("finite data required");mean+=(v-mean)/static_cast<double>(++n);}
  return mean;
}

// Self-test entry point. Returns 1 if any check failed so CTest reports it.
int main(){
  int failures=0;
  // check(ok, name): print PASS/FAIL and count failures.
  auto check=[&](bool ok,const char* name){std::cout<<(ok?"PASS ":"FAIL ")<<name<<'\n';if(!ok)++failures;};
  // Known answer: mean of {2,4,6} is exactly 4 (compare with a tolerance anyway).
  check(std::abs(checked_mean({2,4,6})-4)<1e-12,"known mean");
  // Edge case: a single value is its own mean.
  check(checked_mean({7})==7,"singleton");
  // Contract: empty input must throw invalid_argument, not return 0 or NaN.
  bool threw=false;try{checked_mean({});}catch(const std::invalid_argument&){threw=true;}
  check(threw,"empty input rejected");
  // Property: shifting every value by 10 shifts the mean by exactly 10.
  check(std::abs(checked_mean({12,14,16})-checked_mean({2,4,6})-10)<1e-12,"translation invariant");
  return failures?1:0;
}
