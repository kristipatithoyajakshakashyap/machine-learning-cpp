# Math intuition

P(t|c)=(count(t,c)+alpha)/(total_tokens(c)+alpha*V). log score(c)=log prior(c)+sum_t count(t)*log P(t|c). For class counts [3,1] and alpha=1, token probabilities are [4/6,2/6]. Normalize log scores stably to obtain probabilities.
