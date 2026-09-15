# Math intuition

Recall@10=hits/relevant_test_items. DCG@10=sum(1/log2(rank+1)) for binary-relevant recommendations; NDCG=DCG/ideal_DCG. If one of two relevant items appears at rank 1, recall=1/2 and NDCG=1/(1+1/log2(3)). Evaluate each user then average across eligible users.
