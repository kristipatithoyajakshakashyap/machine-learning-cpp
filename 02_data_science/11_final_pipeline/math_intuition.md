# Mathematical intuition

Median imputation replaces absent x_j with the training median m_j. Standardization is z_j=(x_j-mu_j)/s_j, using population standard deviation and s_j=1 for constant features. PCA centers z again using its fitted mean and projects score_k=sum_j (z_j-pca_mean_j)*loading[k][j]. Never use loading[0][k] inside a sum over j.

If the training class means on PC1 are -2 and 4, the threshold is (-2+4)/2=1. Predict class 1 above 1. If their ordering reverses, the inequality reverses. Accuracy=(TP+TN)/N; precision=TP/(TP+FP); recall=TP/(TP+FN). A zero denominator returns zero here and must be interpreted with the confusion counts.
