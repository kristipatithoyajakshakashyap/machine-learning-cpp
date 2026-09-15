# Math intuition

## Why averaging reduces error

Take M members whose errors on a row have mean zero, variance s^2 and pairwise correlation r. The variance of their average is

    Var(mean) = r s^2 + (1 - r) s^2 / M

The second term vanishes as M grows; the first does not. Averaging removes the independent part of the members' errors but keeps the shared part. Diversity means a small r, which is why three different model families beat three copies of one family.

## Hard voting

Members output labels l_1, ..., l_M. The ensemble returns

    argmax_c  sum_m [l_m == c]

For M = 3 independent members each correct with probability p > 1/2 on a binary task, the majority is correct with probability p^3 + 3 p^2 (1 - p), which exceeds p (for p = 0.8: 0.896). The independence assumption is the fragile part.

## Soft voting

Members output probability rows P_m (length K, sum 1). The ensemble uses

    P(c) = (1/M) sum_m P_m(c)

which still sums to 1, and predicts argmax_c P(c). Because it is a mean of probabilities, a confident member moves the average more than a hesitant one.

## Stacking with a logistic meta-learner

Stack the member rows into one feature vector z of length M K:

    z = [P_1(0..K-1), P_2(0..K-1), P_3(0..K-1)]

The meta-learner is multinomial logistic regression on z:

    P(c | z) = softmax_c (w_c . z + b_c)

Soft voting is the special case w_c = (1/M) e_c for every member block. Stacking searches over all w, so it can do no worse than soft voting on its training matrix; whether it does better on new data depends on the OOF matrix being honest.

## Out-of-fold construction

With folds F_1, ..., F_K of the training rows, and members fitted on all rows except F_k written as P_m^{(-k)}:

    z_i = [P_1^{(-k(i))}(x_i), P_2^{(-k(i))}(x_i), P_3^{(-k(i))}(x_i)]   where i in F_k(i)

Every row's meta-features come from members that did not train on it. The meta-learner is fitted on (z_i, y_i). Then the members are refitted on all training rows to produce z at prediction time. The small mismatch (members at prediction time saw 100% of the training rows, OOF members saw (K - 1)/K of them) is the accepted price; it is far smaller than the in-sample leak it replaces.

## Reading the meta-weights

For meta-class c and member m, the weight on P_m(c) is the "trust" in that member for that class; weights on P_m(c') for c' != c capture how another class's probability from that member shifts the ensemble. With three members and three classes there are 27 weights plus 3 intercepts, fitted on 142 OOF rows; expect them to be noisy, and read only their sign and rough magnitude.
