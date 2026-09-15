#include <iostream>
int main() {
  std::cout
      << R"LESSON(Silhouette compares within-cluster cohesion with separation from the nearest other cluster. Inspect the distribution and cluster profiles, not only one average. It favors compact separated groups and can prefer fewer clusters than an external taxonomy. Compare algorithms using exactly the same observations, distance, and preprocessing. Resampling stability measures sensitivity to which observations are available.)LESSON"
      << '\n';
}
