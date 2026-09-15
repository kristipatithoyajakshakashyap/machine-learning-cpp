// 02_math_intuition.cpp
// Lesson: why feature subsampling decorrelates trees.
// Classification forests draw max_features ~ sqrt(p) features per split.
// The DELIBERATE randomness does two things: any single split is weaker, but
// the trees become less correlated, and averaging many weakly-correlated
// trees beats averaging near-twin trees.  The counts below show how many
// different feature subsets each level can pick from.
// sklearn equivalent: sklearn.ensemble.RandomForestClassifier
// (max_features='sqrt').
//
// EXPECTED OUTPUT:
//   wine p=13 : sqrt(p) = 3.605551  -> max_features ~ 4
//   feature subsets C(13,4) = 715      <- lots of freedom per split
//   iris  p=4  : sqrt(p) = 2          -> C(4,2) = 6
//   p=2        : C(2,1) = 2
//   every tree is likely a genuinely different model

#include <cmath>
#include <iomanip>
#include <iostream>

long comb(long n, long k) {
  if (k > n) return 0;
  long r = 1;
  for (long i = 1; i <= k; ++i) r = r * (n - k + i) / i;
  return r;
}

int main() {
  std::cout << std::setprecision(7);
  std::cout << "wine p=13 : sqrt(p) = " << std::sqrt(13.0)
            << "  -> max_features ~ 4\n";
  std::cout << "feature subsets C(13,4) = " << comb(13, 4)
            << "      <- lots of freedom per split\n";
  std::cout << "iris  p=4  : sqrt(p) = " << std::sqrt(4.0)
            << "          -> C(4,2) = " << comb(4, 2) << "\n";
  std::cout << "p=2        : C(2,1) = " << comb(2, 1) << "\n";
  std::cout << "every tree is likely a genuinely different model\n";
  return 0;
}
