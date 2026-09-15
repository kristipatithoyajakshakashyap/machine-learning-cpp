# Mathematical intuition

a(i) is the mean distance to other members of its cluster. b(i) is the smallest mean distance to another cluster. s(i)=(b-a)/max(a,b). For points 0,2,8,10 and labels 0,0,1,1, scores are 7/9,5/7,5/7,7/9; the mean is 47/63. Singleton samples score zero only in a valid partition. The domain is 2 <= number of labels <= n-1. A zero denominator produces zero.

Work through the four-point fixture by hand before running the numerical lesson. Change one point and explain how the objective or projection changes.
