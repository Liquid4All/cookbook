# Evaluation results

All models are scored the same way: one JSON answer per photo, greedy decoding for LFM3-VL-2B, and one-to-one matching of predicted and annotated boxes at IoU ≥ 0.5. LFM3-VL-2B ran at original image resolution. GPT-6 Luna used its default temperature, with images capped at 1,568 pixels on the longest side.

Checkpoints were chosen on the validation split only. The test split was scored once, after the final model was picked.

## Test split (1,372 photos)

| Model | Typed F1 | Box-only F1 | Precision | Recall | Correct image type | Parse | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LFM3-VL-2B fine-tuned (SFT + GRPO) | 0.106 | 0.130 | 0.121 | 0.094 | 912/1372 | 1372/1372 | 190 / 1374 / 1840 |
| LFM3-VL-2B base | 0.003 | 0.051 | 0.003 | 0.003 | 140/1372 | 1264/1372 | 6 / 2207 / 2024 |
| GPT-6 Luna | 0.010 | 0.058 | 0.009 | 0.012 | 112/1372 | 1372/1372 | 24 / 2724 / 2006 |

"Correct image type" counts photos where the most frequent predicted label is the annotated type. The [README](README.md) reports the same metric with tied labels sharing the credit (66.5%, 9.2% and 8.5%) and 95% bootstrap confidence intervals. Paired differences, fine-tuned minus the other model: type accuracy +57.4 points versus base [+54.7, +60.4] and +58.1 versus GPT-6 Luna [+55.2, +60.9]; detection F1 +0.103 versus base [+0.089, +0.118] and +0.096 versus GPT-6 Luna [+0.081, +0.111].

## Validation split (720 photos)

| Model / checkpoint | Typed F1 | Box-only F1 | Precision | Recall | Correct image type | Parse | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LFM3-VL-2B base | 0.001 | 0.055 | 0.001 | 0.001 | 72/720 | 663/720 | 1 / 1084 / 1036 |
| SFT epoch 1 | 0.071 | 0.088 | 0.062 | 0.084 | 480/720 | 711/720 | 87 / 1314 / 950 |
| SFT epoch 2 | 0.089 | 0.107 | 0.083 | 0.096 | 487/720 | 714/720 | 100 / 1103 / 937 |
| GRPO step 400 | 0.120 | 0.145 | 0.136 | 0.108 | 484/720 | 720/720 | 112 / 710 / 925 |
| GRPO step 800 | 0.104 | 0.122 | 0.121 | 0.092 | 480/720 | 720/720 | 95 / 693 / 942 |
| GPT-6 Luna | 0.009 | 0.057 | 0.008 | 0.011 | 64/720 | 720/720 | 11 / 1365 / 1026 |

SFT epoch 2 and GRPO step 400 are the checkpoints the notebooks continue from and stop at. Box-only F1 ignores the defect type but still requires IoU ≥ 0.5, which is strict for small or loosely drawn boxes. Most of GRPO's gain over SFT comes from fewer extra boxes (1,103 → 710), while recall stays low.

## GRPO training monitor (200 fixed validation photos)

| Checkpoint | Typed F1 | Box-only F1 | Correct image type | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: |
| SFT start | 0.083 | 0.093 | 129/200 | 24 / 283 / 247 |
| GRPO step 400 | 0.111 | 0.131 | 123/200 | 28 / 206 / 243 |
| GRPO step 800 | 0.091 | 0.108 | 121/200 | 22 / 190 / 249 |

The monitor is a smaller sample than the 720-photo table above, so compare its rows with each other, not with that table. Training past step 400 lowered detection F1 on both.

## Scorecard files

- [`results/full_test_scorecard.csv`](results/full_test_scorecard.csv): the test split.
- [`results/full_validation_scorecard.csv`](results/full_validation_scorecard.csv): the validation split.
- [`results/grpo_monitor_scorecard.csv`](results/grpo_monitor_scorecard.csv): the GRPO monitor.

Typed F1 is `2*TP / (2*TP + FP + FN)`.
