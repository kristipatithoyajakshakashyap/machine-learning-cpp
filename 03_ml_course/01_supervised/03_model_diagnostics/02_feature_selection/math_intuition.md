# Math intuition

## ANOVA F for one column

Let a column have values x_i with class labels y_i in {1..k}, class means m_c, class sizes n_c and grand mean m. Then

    between = sum_c n_c (m_c - m)^2            (k - 1 degrees of freedom)
    within  = sum_i (x_i - m_{y_i})^2           (n - k degrees of freedom)
    F       = (between / (k - 1)) / (within / (n - k))

F is large when the class means are far apart compared with the scatter inside each class. In `02_math_intuition.cpp` column 0 has class means 2 and 8 around a grand mean of 5, so between = 3*9 + 3*9 = 54, within = 4, and F = 54. Column 1 has identical class means, so between = 0 and F = 0 no matter how large its variance is. A constant column has within = 0 and between = 0 and is scored 0.

## |Pearson r| for a continuous target

    r = sum (x_i - mx)(y_i - my) / sqrt(sum (x_i - mx)^2 * sum (y_i - my)^2)

The absolute value is used because a strongly negative correlation is as useful as a strongly positive one.

## Greedy forward selection

With chosen set S the score of a candidate j is CV(S + {j}). The step adds argmax_j CV(S + {j}). This is a greedy ascent over subsets: it never revisits a choice, so a column that was best on its own may stay even after a later column made it redundant. Its strength is that CV is computed with the real model, so interactions count.

## Lasso path

The lasso objective on standardized columns z is

    (1 / 2n) ||y - b - Z w||^2 + lambda ||w||_1

Coordinate descent updates one weight at a time with soft thresholding: w_j = S(rho_j, lambda) / (z_j . z_j / n), where S(a, t) = sign(a) max(|a| - t, 0). A weight stays at zero while |rho_j| <= lambda, so as lambda decreases features enter in order of how strongly they explain the residual of the ones already in. `l1_scores` ranks each feature by the largest lambda at which its weight is non-zero.

## Why the three rankings differ

The filter measures marginal separation; the lasso measures conditional (linear) contribution given the other columns; the wrapper measures conditional contribution given the forest's non-linear view. Breast cancer has strongly correlated "mean/worst" pairs, so the filter ranks both members of a pair high while lasso and forward selection keep one and drop the other. `03_implementation` plots filter rank against lasso rank so you can see that disagreement.
