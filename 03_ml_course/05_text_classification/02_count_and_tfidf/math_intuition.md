# Math intuition

idf(t)=log((1+N)/(1+df(t)))+1. For N=3,df=1, idf=1+log(2). Multiply counts by IDF then L2-normalize each document. Empty or all-unknown documents remain zero vectors.
