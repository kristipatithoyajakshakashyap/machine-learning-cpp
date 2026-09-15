#include <iostream>
int main() {
  std::cout
      << R"LESSON(N_eps(x)={y:||x-y||<=eps}. Core iff |N_eps(x)|>=min_samples. At eps=2.1 and min_samples=2, points 0,2,8,10 make two dense pairs; an added point 30 is noise. Core connectivity is transitive; border membership can depend on traversal when it touches two components. The implementation resolves such ties by input order.)LESSON"
      << '\n';
}
