# Pipeline theory

A pipeline is a composition of transformations with learned state. Splitting must precede learning that state: even an unlabeled test median leaks information about the future distribution. Exploratory plots use the training partition.

PCA finds orthogonal directions of greatest feature variance. A high-variance direction need not separate survivors. Our transparent classifier compares the first principal-component score against the midpoint of the two training class means. The majority classifier gives a necessary performance reference.
