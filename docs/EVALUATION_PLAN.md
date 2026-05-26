# Evaluation Plan — Industrial: Models that learn how processes unfold

## Main metric
Next-step prediction accuracy/MAE, anomaly F1/AUC, early warning lead time, robust performance under noise

## Required comparisons
- naive baseline;
- simple ML baseline;
- improved model/layer;
- stress test or edge case.

## Judge-facing proof
Create one chart or table showing:
- baseline result;
- improved result;
- business interpretation;
- limitation.

## Anti-cheating / reliability checks
- no train-test leakage;
- fixed random seed;
- clear synthetic vs real label;
- saved split files;
- repeatable command.
