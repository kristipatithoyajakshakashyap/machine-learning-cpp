#include <iostream>
int main() {
  std::cout
      << R"LESSON(Expected unsuccessful-search path c(n)=2H_(n-1)-2(n-1)/n, with c(1)=0 and c(2)=1. Score s(x)=2^(-E[h(x)]/c(psi)). If average path equals c(psi), score=0.5; paths half that long give sqrt(0.5), about 0.707. Unsplit leaves add c(leaf_size) to path length.)LESSON"
      << '\n';
}
