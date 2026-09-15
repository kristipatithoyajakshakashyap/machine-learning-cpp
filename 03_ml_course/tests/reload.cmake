execute_process(COMMAND "${PREDICT}" --predict "${RESULTS}/quick/data/holdout_features.csv"
  --model "${RESULTS}/quick/model" RESULT_VARIABLE status)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Fresh-process inference failed: ${status}")
endif()
execute_process(COMMAND "${VERIFY}" "${RESULTS}/quick/evaluation/predictions.csv"
  "${PREDICT_RESULTS}/predictions.csv" RESULT_VARIABLE status)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Fresh-process predictions did not match: ${status}")
endif()
