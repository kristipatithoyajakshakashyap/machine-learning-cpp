#include <iostream>
int main() {
  std::cout
      << R"LESSON(p(x)=sum_c pi_c N(x|mu_c,diag(v_c)). r_ic=pi_c N_c(x_i)/sum_l pi_l N_l(x_i). N_c=sum_i r_ic; mu_c=sum_i r_ic x_i/N_c; v_c=sum_i r_ic(x_i-mu_c)^2/N_c+lambda. BIC=-2 logL+q log n; AIC=-2 logL+2q; q=2kp+k-1 for diagonal covariance. For equal priors and variances at means 0 and 10, x=5 has responsibilities (0.5,0.5).)LESSON"
      << '\n';
}
