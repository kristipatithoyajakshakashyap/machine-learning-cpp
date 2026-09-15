# Titanic pipeline: stage 3

Predict recorded survival using passenger attributes. PassengerId is an identifier, not a feature.

Rows: 891; training: 712; hold-out: 179. Split seed 42.

Age median 28 and Fare median 14.427099999999999 are learned from training only. Cabin absence is an empty string or NA.

The model is a nearest-class-mean threshold on PC1, not logistic regression. PCA optimizes feature variance, not predictive accuracy. No hyperparameters are selected on test.

Training numeric summary:
```
            PassengerId  Survived  Pclass  Age  SibSp  Parch  Fare  family_size  is_child  has_cabin  Sex_male  
count               712       712     712  712    712    712   712          712       712        712       712  
mean            445.823  0.383427  2.307584  29.16562  0.5210674  0.377809  31.85403     1.898876  0.0758427  0.2289326  0.6502809  
std            257.9633  0.4865627  0.8366088  13.16162  1.12489  0.8002895  50.87421     1.628128  0.2649324  0.4204411  0.4772164  
min                   1         0       1  0.67      0      0     0            1         0          0         0  
25%              220.75         0       2   22      0      0  7.9177            1         0          0         0  
50%               446.5         0       3   28      0      0  14.4271            1         0          0         1  
75%               668.5         1       3   35      1      0  30.5            2         0          0         1  
max                 891         1       3   80      8      6  512.3292           11         1          1         1  
```

Historical observational associations do not establish causality; family groups can cross this educational stratified split. A deployment study should evaluate family-group hold-outs.
