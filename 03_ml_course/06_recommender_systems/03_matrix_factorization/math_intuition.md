# Math intuition

rating_hat=mu+b_u+b_i+p_u dot q_i. For residual e=r-rating_hat, SGD updates b_u += eta*(e-lambda*b_u), p_u += eta*(e*q_i-lambda*p_u), with the old p_u used for the paired q_i update. The implementation penalizes per observed rating and shuffles only the training interactions.
