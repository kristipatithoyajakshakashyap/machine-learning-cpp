# SMS spam classification

Held-out F1: 0.746411; precision: 1; recall: 0.59542. Selected parameter 0.0001 with mean training CV F1 0.692998.

Exact duplicate messages were removed before splitting. Vocabulary and IDF were learned separately within each fold. The cap retains frequent training terms; unknown words are ignored and an all-unknown message uses the model intercept/prior. The ASCII tokenizer does not perform Unicode normalization. Source collection dates and sender identities are unavailable; random evaluation does not establish future or sender-independent performance. Inspect false positives before deployment; flagging legitimate SMS has a real user cost. EDA summarizes training rows only. Calibration and threshold tables describe the held-out set; do not tune a deployment threshold on it.
