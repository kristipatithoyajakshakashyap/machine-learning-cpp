#include <iostream>
int main() {
  std::cout << R"LESSON(# Mathematical intuition

J = sum_i ||x_i - mu_(z_i)||^2. Holding assignments fixed, dJ/dmu_c = 2 sum_(i:z_i=c)(mu_c-x_i), so mu_c is the member mean. For points 0, 2, 8, 10 and two groups {0,2}, {8,10}, centers are 1 and 9; inertia is 1+1+1+1=4. Adding clusters cannot increase the globally optimal inertia, so an elbow is a heuristic, not proof of the true count.

Work through the four-point fixture by hand before running the numerical lesson. Change one point and explain how the objective or projection changes.
)LESSON";
}
