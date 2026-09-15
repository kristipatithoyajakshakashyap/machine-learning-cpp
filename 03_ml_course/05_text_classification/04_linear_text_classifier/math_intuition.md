# Math intuition

p(spam|x)=sigmoid(w dot x+b). Minimize mean binary cross entropy+lambda*||w||^2/2. The gradient for feature j is mean((p-y)*x_j)+lambda*w_j. The bias is unpenalized. The batch step is the reciprocal of a Hessian upper bound: 0.25*(1+max row squared norm)+lambda, including the bias coordinate.
