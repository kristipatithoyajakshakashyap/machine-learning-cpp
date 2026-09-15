# Implementation

The shared track Workflow.hpp coordinates stratified holdout, training-only five-fold F1 selection, preprocessing, model fit, metrics, vocabulary persistence and reload. Canonical implementations live in the numbered preprocessing, vectorization, NB and linear classifier modules. Sparse rows store only nonzero (index,value) pairs. The training corpus never reads test documents while building vocabulary or IDF.
