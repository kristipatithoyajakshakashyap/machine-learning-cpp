# Math intuition

rating_hat_i=(sum_ratings_i+10*global_mean)/(count_i+10). Ranking score is count_i. For count=2,sum=10,global_mean=3, the smoothed prediction is 40/12=3.333. Exclude items the user already rated.
