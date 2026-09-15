# Isolation Forest

Isolation Forest isolates observations by random feature/threshold splits. Rare or extreme points usually need fewer splits. Average path length across subsampled trees becomes an anomaly score. The score describes unusual feature values, not fraud or data error. Choose a threshold using a validation review budget or labeled validation examples, never final test labels.
