# Mathematical intuition

N_eps(x)={y:||x-y||<=eps}. Core iff |N_eps(x)|>=min_samples. At eps=2.1 and min_samples=2, points 0,2,8,10 make two dense pairs; an added point 30 is noise. Core connectivity is transitive; border membership can depend on traversal when it touches two components. The implementation resolves such ties by input order.

Work through the four-point fixture by hand before running the numerical lesson. Change one point and explain how the objective or projection changes.
