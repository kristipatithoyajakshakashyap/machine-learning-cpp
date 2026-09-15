#include <iostream>
int main() {
  std::cout << R"LESSON(# Mathematical intuition

Single(A,B)=min d(a,b); complete=max d(a,b); average=sum d(a,b)/(|A||B|). Ward delta SSE=|A||B|/(|A|+|B|)*||mean(A)-mean(B)||^2; reported linkage height=sqrt(2 delta SSE). On 0,2,8,10 the first merges have height 2, then the Ward root has height sqrt(128). A cut at two clusters recovers {0,2} and {8,10}.

Work through the four-point fixture by hand before running the numerical lesson. Change one point and explain how the objective or projection changes.
)LESSON";
}
