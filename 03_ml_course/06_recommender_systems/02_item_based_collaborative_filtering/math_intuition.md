# Math intuition

sim(i,j)=dot(centered_item_i,centered_item_j)/(norm_i*norm_j), multiplied by overlap/(overlap+10). Retain the K most similar positive neighbors globally per item. prediction=user_mean+sum(sim*centered_rating)/sum(sim) over those neighbors the user rated; fall back to smoothed item mean if none apply.
