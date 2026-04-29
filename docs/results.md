# Results

This file records concrete experiment outputs from the current `charstreamer`
iteration work.

## Full-Corpus Legacy Docking

Source reports:

- `/tmp/charboundary-small-python-rf-full-report.json`
- `/tmp/charboundary-small-full-rust-metric.json`

| experiment | train_rows | positives | SCOTUS precision | SCOTUS recall | SCOTUS F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| original_python_charboundary_small | 554173 | 156815 | 0.9293 | 0.4040 | 0.5632 |
| charstreamer_exact_legacy_python_rf | 555873 | 156851 | 0.9149 | 0.3937 | 0.5505 |

Interpretation:

- the old `0.86` SCOTUS number came from a different evaluation contract; under the current boundary-position metric, the original Python small model is about `0.563`
- after fixing ALEA label semantics, terminal candidate scanning, and legacy sampling behavior, the exact-legacy `charstreamer` Python RF path docks closely to the original Python baseline
- future quality claims should cite the evaluation contract explicitly, because the absolute numbers move a lot between evaluators

## Full-Corpus Rust Tree Search

Source reports:

- `/tmp/current-legal-tree-local-structure-full-report.json`
- `/tmp/current-legal-tree-directional-class-counts-full-report.json`
- `/tmp/current-legal-tree-structure-directional-class-counts-full-report.json`
- `/tmp/charboundary-small-tree-full-report.json`
- `/tmp/charboundary-small-tree-hybrid-structure-full-report.json`

| experiment | train_rows | validation F1 | SCOTUS F1 | SCOTUS cps |
| --- | ---: | ---: | ---: | ---: |
| charboundary_small_tree_full | 555661 | 0.0000 | 0.5279 | 9538863.6 |
| charboundary_small_tree_hybrid_structure_full | 499437 | 0.8365 | 0.5574 | 9303055.3 |
| current_legal_tree_local_structure_full | 420455 | 0.8369 | 0.5582 | 1183138.1 |
| current_legal_tree_directional_class_counts_full | 420455 | 0.8373 | 0.5630 | 1169490.9 |
| current_legal_tree_structure_directional_class_counts_full | 420455 | 0.8379 | 0.5616 | 1166046.6 |

Interpretation:

- the pure legacy Rust tree is no longer the target; current feature stacks beat it clearly on SCOTUS
- the first hybrid legacy+structure tree is competitive, but it does not beat the best current tree
- within this initial full-corpus tree search, `current_legal_tree_directional_class_counts_full` was the strongest same-window directional baseline and became the anchor for the later window/radius follow-up sweeps
- under the current boundary-position metric, that directional-count baseline essentially matches the original Python small model on SCOTUS
- deeper and entropy-tuned variants of the directional tree both regress on SCOTUS, so the plain directional preset is also the best-tuned version found so far

## Full-Corpus Feature Sweep

Source reports:

- `/tmp/current-legal-feature-sweep-unicode.md`
- `/tmp/current-legal-window-radius-sweep.md`

| experiment | validation F1 | candidate F1 | validation cps | train_s | SCOTUS F1 | SCOTUS cps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_legal_tree_directional_class_counts_full | 0.8373 | 0.9536 | 34512969.7 | 45.465 | 0.5630 | 1169421.6 |
| current_legal_tree_directional_class_counts_window_3_1_full | 0.8369 | 0.9537 | 36603557.4 | 33.572 | 0.5639 | 1178220.6 |
| current_legal_tree_directional_class_counts_window_3_3_full | 0.8380 | 0.9553 | 35195016.2 | 42.025 | 0.5643 | 1170422.1 |
| current_legal_tree_directional_class_counts_window_5_5_full | 0.8380 | 0.9553 | 34178737.3 | 50.445 | 0.5628 | 1175782.2 |
| current_legal_tree_directional_class_counts_window_7_5_full | 0.8381 | 0.9554 | 33417693.0 | 52.684 | 0.5635 | 1173993.7 |
| current_legal_tree_directional_unicode_groups_full | 0.8370 | 0.9526 | 27732719.2 | 52.207 | 0.5593 | 1163958.8 |
| current_legal_tree_directional_byte_plus_unicode_groups_full | 0.8386 | 0.9566 | 22127449.4 | 58.460 | 0.5613 | 1161266.8 |
| current_legal_tree_directional_unicode_categories_full | 0.8383 | 0.9559 | 26162366.7 | 62.947 | 0.5611 | 1161707.8 |

Interpretation:

- the current feature family does benefit from a shorter encoded window; `3/1` strictly improves over the old `5/3` full-corpus baseline on both SCOTUS F1 and single-thread throughput
- `3/3` is the highest-SCOTUS-F1 variant in this sweep, but it is slightly slower than `3/1`; both belong on the current Pareto frontier
- larger encoded windows (`5/5`, `7/5`) do not help enough to justify their added cost
- ICU-based Unicode category and category-group counts are now implemented and configurable, but on the current legal corpora they do not beat the byte/ASCII directional-count baseline
- stacking byte directional counts with Unicode groups improves validation metrics while still losing on SCOTUS, so it currently looks like overfitting rather than a real generalization gain

## Full-Corpus Window and Radius Follow-Up

Source reports:

- `/tmp/current-legal-window-radius-sweep.md`

| experiment | validation F1 | candidate F1 | validation cps | train_s | SCOTUS F1 | SCOTUS cps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_legal_tree_directional_class_counts_window_3_1_full | 0.8369 | 0.9537 | 36381514.8 | 33.594 | 0.5639 | 1170311.7 |
| current_legal_tree_directional_class_counts_window_3_3_full | 0.8380 | 0.9553 | 35253958.3 | 42.274 | 0.5643 | 1176560.4 |
| current_legal_tree_directional_class_counts_window_7_7_full | 0.8382 | 0.9556 | 32416864.1 | 55.617 | 0.5621 | 1172810.0 |
| current_legal_tree_directional_class_counts_window_9_7_full | 0.8383 | 0.9558 | 32056838.2 | 57.650 | 0.5630 | 1172190.5 |
| current_legal_tree_directional_class_counts_window_3_1_radius_6_full | 0.8361 | 0.9529 | 42564793.8 | 28.238 | 0.5569 | 1179631.2 |
| current_legal_tree_directional_class_counts_window_3_1_radius_24_full | 0.8359 | 0.9509 | 28500674.3 | 42.106 | 0.5599 | 1169487.0 |
| current_legal_tree_directional_class_counts_window_3_1_radius_36_full | 0.8370 | 0.9533 | 23590656.5 | 47.010 | 0.5633 | 1154250.2 |

Interpretation:

- pushing the encoded window larger (`7/7`, `9/7`) improves validation metrics a little but does not improve SCOTUS F1; those larger windows are slower and stay off the Pareto frontier
- the shorter encoded-window conclusion holds under the broader sweep: `3/1` remains the best default and `3/3` remains the best quality-oriented neighboring point
- shrinking the directional count radius too far (`6/6`) hurts SCOTUS quality even though it improves validation throughput, so the count block is carrying real generalization signal
- expanding the directional count radius beyond the current default does not pay off; `36/36` nearly recovers baseline SCOTUS quality but is materially slower, while `24/24` regresses on both quality and training cost
- the current sweet spot still looks like a short encoded window plus a moderate directional-count radius, not a large encoded window or a very broad context count

## Archived Full-Corpus Alternative Model Sweep

These results are retained as historical comparison data. The SmartCore and
Linfa backend crates and runnable specs have since been removed from the active
workspace; Burn is the only external Rust model backend.

Source reports:

- `/tmp/current-legal-tree-directional-class-counts-full-report.json`
- `/tmp/current-legal-smartcore-logistic-directional-full-report.json`
- `/tmp/current-legal-linfa-logistic-directional-full-report.json`
- `/tmp/current-legal-rf-directional-class-counts-small-full-report.json`

| experiment | train_s | validation F1 | validation cps | SCOTUS F1 | SCOTUS cps |
| --- | ---: | ---: | ---: | ---: | ---: |
| current_legal_tree_directional_class_counts_full | 52.559 | 0.8373 | 34385301.0 | 0.5630 | 1169490.9 |
| current_legal_smartcore_logistic_directional_full | 114.644 | 0.8174 | 38163890.1 | 0.5197 | 1167190.6 |
| current_legal_linfa_logistic_directional_full | 60.628 | 0.3621 | 38473188.7 | 0.3468 | 1181264.4 |
| current_legal_rf_directional_class_counts_small_full | 271.729 | 0.8356 | 4037693.7 | 0.5431 | 951667.5 |

Operational failures and rejects:

- `current_legal_smartcore_knn_directional_full`: single-thread fit stayed CPU-bound for minutes on the full `420k`-row training matrix and was dropped before completion; it is not a practical Pareto candidate
- `current_legal_smartcore_gaussian_nb_directional_full`: full-corpus release run aborts with a `smartcore` stack overflow, even with `RUST_MIN_STACK=33554432`

Interpretation:

- none of the newly tested alternative backends pareto-dominate the current directional tree on both SCOTUS F1 and single-thread throughput
- `smartcore` logistic keeps similar inference speed, but loses too much cross-domain F1
- `linfa` logistic is clearly the wrong fit on the current feature stack
- the small directional RF is strong on validation, but it is slower than the tree and still worse on SCOTUS
- at that point, `current_legal_tree_directional_class_counts_window_3_1_full` remained the best full-corpus single-thread default among the archived non-Burn alternatives

## Full-Corpus Burn Neural Sweep

Source reports:

- `/tmp/current-legal-tree-directional-class-counts-full-report.json`
- `/tmp/current-legal-burn-shallow-mlp-directional-full-report.json`
- `/tmp/current-legal-burn-deep-mlp-directional-full-report.json`
- `/tmp/current-legal-burn-window-cnn-directional-full-report.json`
- `/tmp/current-legal-burn-window-gru-directional-full-report.json`
- `/tmp/current-legal-burn-window-lstm-directional-full-report.json`

| experiment | train_s | validation F1 | validation cps | SCOTUS F1 | SCOTUS cps |
| --- | ---: | ---: | ---: | ---: | ---: |
| current_legal_tree_directional_class_counts_full | 52.559 | 0.8373 | 34385301.0 | 0.5630 | 1169490.9 |
| current_legal_burn_shallow_mlp_directional_full | 17.135 | 0.8357 | 10224306.3 | 0.5478 | 1157583.0 |
| current_legal_burn_deep_mlp_directional_full | 42.713 | 0.8357 | 5976142.6 | 0.5485 | 1149156.0 |
| current_legal_burn_window_cnn_directional_full | 31.347 | 0.8331 | 5985539.6 | 0.5596 | 1122696.6 |
| current_legal_burn_window_gru_directional_full | 219.067 | 0.8381 | 544983.9 | 0.5589 | 892954.4 |
| current_legal_burn_window_lstm_directional_full | 317.880 | 0.8382 | 398607.9 | 0.5613 | 803631.0 |

Interpretation:

- no tested Burn neural model pareto-dominates the current directional tree on both SCOTUS F1 and single-thread throughput
- the shallow Burn MLP is the most practical neural baseline: it trains quickly and keeps throughput near the tree, but it gives up too much SCOTUS F1
- the window CNN is the best Burn quality/speed compromise, but it still trails the tree slightly on both axes
- the GRU and LSTM slightly improve over the MLPs on SCOTUS, but their single-thread throughput is dramatically worse
- on this task and feature stack, Burn neural models are competitive enough to keep as configurable options, but not strong enough to replace the current tree default

## Archived Candle Neural Sweep

Source reports:

- `/tmp/current-legal-tree-directional-class-counts-full-report.json`
- `/tmp/current-legal-candle-shallow-mlp-directional-full-report.json`
- `/tmp/current-legal-candle-shallow-mlp-directional-wide-full-report.json`
- `/tmp/current-legal-candle-deep-mlp-directional-full-report.json`
- `/tmp/current-legal-candle-deep-mlp-directional-wide-full-report.json`
- `/tmp/current-legal-candle-window-cnn-directional-full-report.json`
- `/tmp/current-legal-candle-window-cnn-directional-wide-full-report.json`
- `/tmp/current-legal-candle-window-gru-directional-full-report.json`
- `/tmp/current-legal-candle-window-gru-directional-lite-full-report.json`
- `/tmp/current-legal-candle-window-lstm-directional-full-report.json`
- `/tmp/current-legal-candle-window-lstm-directional-lite-full-report.json`

| experiment | train_s | validation F1 | validation cps | SCOTUS F1 | SCOTUS cps |
| --- | ---: | ---: | ---: | ---: | ---: |
| current_legal_tree_directional_class_counts_full | 52.288 | 0.8373 | 34581439.7 | 0.5630 | 1176818.7 |
| current_legal_candle_shallow_mlp_directional_full | 12.953 | 0.8362 | 16641302.7 | 0.5511 | 1159788.4 |
| current_legal_candle_shallow_mlp_directional_wide_full | 37.611 | 0.8354 | 12409938.2 | 0.5503 | 1143434.2 |
| current_legal_candle_deep_mlp_directional_full | 33.423 | 0.8362 | 9700602.8 | 0.5526 | 1145070.0 |
| current_legal_candle_deep_mlp_directional_wide_full | 63.663 | 0.8361 | 7094819.1 | 0.5479 | 1119846.6 |
| current_legal_candle_window_cnn_directional_full | 33.751 | 0.8356 | 10385542.2 | 0.5490 | 1146230.5 |
| current_legal_candle_window_cnn_directional_wide_full | 87.742 | 0.8349 | 6696912.9 | 0.5475 | 1110851.2 |
| current_legal_candle_window_gru_directional_full | 120.842 | 0.8388 | 1551847.8 | 0.5605 | 980640.8 |
| current_legal_candle_window_gru_directional_lite_full | 67.572 | 0.8374 | 1688630.6 | 0.5580 | 1014648.1 |
| current_legal_candle_window_lstm_directional_full | 144.867 | 0.8387 | 1612185.0 | 0.5610 | 958122.5 |
| current_legal_candle_window_lstm_directional_lite_full | 76.283 | 0.8371 | 1841829.7 | 0.5569 | 1015352.9 |

Interpretation:

- no tested Candle neural model pareto-dominates the current directional tree on both SCOTUS F1 and single-thread throughput
- the shallow Candle MLP is the most practical Candle baseline: it trains quickly and keeps SCOTUS throughput near the tree, but it loses too much F1
- wider MLP and CNN variants do not help on this feature stack; they reduce throughput and slightly reduce SCOTUS F1
- the recurrent Candle models are the strongest on SCOTUS quality, but even the lighter GRU/LSTM variants remain too slow to replace the tree as the default
- Candle has been removed from the active build and experiment spec surface; keep these numbers only as historical reference

## Synthetic Semantic Span Burn Slice

Source data:

- `data/synthetic/kl3m_streaming_spans_20260428_per_label_snapshot_2962.jsonl`
- snapshot rows: `2,963`
- source mix: corrected `1k` per-label file plus current live `per_label_20260428_4k` shard rows

Source reports:

- `/tmp/charstreamer-synthetic-burn-2963-full.json`
- `/tmp/charstreamer-synthetic-burn-2963-inside.json`
- `/tmp/charstreamer-synthetic-burn-2963-inside-sqrt-balanced.json`
- `/tmp/charstreamer-synthetic-burn-2963-full-sqrt-balanced.json`

Configuration:

- model: Burn CPU `NdArray` MLP
- features: encoded byte window `7/7`, ASCII neighbor classes, directional byte-class counts radius `24`, directional Unicode group counts radius `24`, line-structure byte counts
- split: document-level `80/20`
- training positions: UTF-8 boundary positions; train negatives sampled at `0.5`; validation uses all positions

| experiment | outputs | train_rows | valid_rows | train_s | valid rows/s | macro F1 | inside F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| burn_full_inside_start_end | 18 | 454730 | 114799 | 25.460 | 563211.5 | 0.2888 | 0.5127 |
| burn_inside_only | 6 | 454110 | 114799 | 31.781 | 582069.0 | 0.5220 | 0.5220 |
| burn_inside_only_sqrt_balanced | 6 | 454110 | 114799 | 30.562 | 615712.9 | 0.5226 | 0.5226 |
| burn_full_sqrt_balanced | 18 | 454730 | 114799 | 23.892 | 594482.2 | 0.2393 | 0.4728 |

Inside-only per-label F1:

| label | unweighted F1 | sqrt-balanced F1 | validation positives |
| --- | ---: | ---: | ---: |
| sentence | 0.8847 | 0.8832 | 70008 |
| paragraph | 0.9037 | 0.9023 | 77901 |
| section | 0.3969 | 0.4023 | 9772 |
| dialogue | 0.0382 | 0.0347 | 1333 |
| list_item | 0.1994 | 0.2057 | 9680 |
| metadata | 0.7093 | 0.7071 | 29518 |

Interpretation:

- the new Burn semantic trainer works end-to-end on corrected per-label synthetic span data
- sentence and paragraph `inside` labels are strong, and metadata is usable
- section/list/dialogue remain weak; this looks like a data separability and supervision issue, not just loss weighting
- `sqrt-balanced` output weighting does not materially improve macro F1 and hurts the full `inside/start/end` model
- rare `start/end` heads should not share the same model/loss naively; next iterations should use boundary-specific sampling, positive-weighted BCE/focal loss, or separate start/end models

## Synthetic Semantic Span Burn Architecture Sweep

Source data:

- `data/synthetic/kl3m_streaming_spans_20260428_per_label_snapshot_2962.jsonl`
- snapshot rows: `2,963`
- split: same document-level `80/20`, seed `19`
- task: `inside` labels only
- environment: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`

Source reports:

- `/tmp/charstreamer-synthetic-burn-sweep-linear.json`
- `/tmp/charstreamer-synthetic-burn-sweep-mlp1_32.json`
- `/tmp/charstreamer-synthetic-burn-sweep-mlp1_64.json`
- `/tmp/charstreamer-synthetic-burn-sweep-mlp1_192.json`
- `/tmp/charstreamer-synthetic-burn-sweep-mlp2_64_32.json`
- `/tmp/charstreamer-synthetic-burn-sweep-mlp2_128_64.json`
- `/tmp/charstreamer-synthetic-burn-sweep-mlp2_192_64.json`
- `/tmp/charstreamer-synthetic-burn-sweep-mlp3_256_128_64.json`

All runs used the same feature stack as the first Burn semantic slice:
encoded byte window `7/7`, ASCII neighbor classes, directional byte-class
counts radius `24`, directional Unicode group counts radius `24`, and
line-structure byte counts. Validation rows/sec is model prediction throughput;
end-to-end chars/sec includes validation feature extraction plus model
prediction.

| experiment | params | train_s | pred rows/s | e2e chars/s | macro P | macro R | macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | linear `1/1/1` | 6.605 | 2310410.2 | 394973.5 | 0.4680 | 0.7849 | 0.5065 |
| mlp1_32 | mlp1 `32/16/8` | 10.216 | 1809076.2 | 378089.4 | 0.4845 | 0.6327 | 0.5193 |
| mlp1_64 | mlp1 `64/32/16` | 17.641 | 1094536.0 | 333455.8 | 0.4969 | 0.5823 | 0.5217 |
| mlp1_192 | mlp1 `192/64/32` | 32.739 | 483944.6 | 240323.5 | 0.4865 | 0.6442 | 0.5218 |
| mlp2_64_32 | mlp2 `64/32/16` | 22.413 | 1004558.8 | 324829.7 | 0.4984 | 0.6576 | 0.5272 |
| mlp2_128_64 | mlp2 `128/64/32` | 37.904 | 543656.7 | 255030.1 | 0.5047 | 0.6011 | 0.5297 |
| mlp2_192_64 | mlp2 `192/64/32` | 44.972 | 406194.4 | 219729.8 | 0.5017 | 0.6499 | 0.5289 |
| mlp3_256_128_64 | mlp3 `256/128/64` | 80.516 | 195900.0 | 139017.2 | 0.5004 | 0.6414 | 0.5285 |

Per-label precision/recall/F1 for the most relevant candidates:

| model | label | precision | recall | F1 |
| --- | --- | ---: | ---: | ---: |
| linear | sentence | 0.8140 | 0.9350 | 0.8703 |
| linear | paragraph | 0.8598 | 0.9286 | 0.8929 |
| linear | section | 0.4111 | 0.3958 | 0.4033 |
| linear | dialogue | 0.0144 | 0.9640 | 0.0283 |
| linear | list_item | 0.1081 | 0.7664 | 0.1894 |
| linear | metadata | 0.6006 | 0.7193 | 0.6546 |
| mlp1_64 | sentence | 0.8503 | 0.9191 | 0.8834 |
| mlp1_64 | paragraph | 0.8734 | 0.9332 | 0.9023 |
| mlp1_64 | section | 0.4048 | 0.4017 | 0.4032 |
| mlp1_64 | dialogue | 0.0176 | 0.2318 | 0.0327 |
| mlp1_64 | list_item | 0.1608 | 0.2584 | 0.1982 |
| mlp1_64 | metadata | 0.6747 | 0.7497 | 0.7102 |
| mlp2_64_32 | sentence | 0.8530 | 0.9255 | 0.8878 |
| mlp2_64_32 | paragraph | 0.8806 | 0.9331 | 0.9061 |
| mlp2_64_32 | section | 0.3992 | 0.4217 | 0.4102 |
| mlp2_64_32 | dialogue | 0.0193 | 0.6174 | 0.0374 |
| mlp2_64_32 | list_item | 0.1581 | 0.2875 | 0.2040 |
| mlp2_64_32 | metadata | 0.6798 | 0.7604 | 0.7179 |
| mlp2_128_64 | sentence | 0.8589 | 0.9193 | 0.8881 |
| mlp2_128_64 | paragraph | 0.8737 | 0.9390 | 0.9052 |
| mlp2_128_64 | section | 0.3902 | 0.4286 | 0.4085 |
| mlp2_128_64 | dialogue | 0.0211 | 0.3226 | 0.0396 |
| mlp2_128_64 | list_item | 0.1877 | 0.2219 | 0.2034 |
| mlp2_128_64 | metadata | 0.6965 | 0.7751 | 0.7337 |

Interpretation:

- the linear model is a strong speed baseline and proves the current features are carrying real signal
- `mlp1_64` is the best simple default if throughput matters; it gains `+0.0152` macro F1 over linear while keeping model prediction above `1M` rows/sec
- `mlp2_64_32` is the current Pareto default candidate; it improves macro F1 to `0.5272` with only a modest throughput hit versus `mlp1_64`
- `mlp2_128_64` has the best macro F1 at `0.5297`, but its speed/quality gain over `mlp2_64_32` is small
- deeper/wider models are not worth it on this dataset; `mlp3_256_128_64` is slower and does not beat the best two-layer model
- dialogue/list-item quality remains data-limited; architecture depth does not fix the weak supervision problem

## Synthetic Semantic Span Burn 5k Refresh

Source data:

- `data/synthetic/kl3m_streaming_spans_20260429_per_label_4k.jsonl`
- `data/synthetic/kl3m_streaming_spans_20260429_per_label_5k.jsonl`
- `data/synthetic/kl3m_streaming_spans_20260429_per_label_4k.summary.json`
- `data/synthetic/kl3m_streaming_spans_20260429_per_label_5k.summary.json`

Generation status:

- four live shards completed at `1,000` accepted rows each
- merged 4k rows: `3,999` unique rows after removing `1` duplicate
- combined corrected 1k + 4k rows: `4,999` unique rows after removing `1` duplicate
- 5k span counts: `sentence=4420`, `paragraph=3578`, `metadata=1941`, `section=961`, `list_item=752`, `dialogue=73`
- 5k shape: `166` empty rows, `3,791` multi-label rows, `3,123` sentence+paragraph rows

Source reports:

- `/tmp/charstreamer-synthetic-burn-5k-inside-mlp2_64_32.json`
- `/tmp/charstreamer-synthetic-burn-5k-full-mlp2_64_32.json`

| experiment | outputs | train_rows | valid_rows | train_s | pred rows/s | macro F1 | inside F1 | start F1 | end F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5k_inside_mlp2_64_32 | 6 | 754830 | 201767 | 37.773 | 1071742.5 | 0.5405 | 0.5405 | - | - |
| 5k_full_mlp2_64_32 | 18 | 755853 | 201767 | 30.740 | 908699.8 | 0.2493 | 0.4737 | 0.1511 | 0.1232 |

Inside-only 5k per-label metrics:

| label | precision | recall | F1 | validation positives |
| --- | ---: | ---: | ---: | ---: |
| sentence | 0.8835 | 0.9474 | 0.9143 | 132730 |
| paragraph | 0.8904 | 0.9484 | 0.9185 | 142103 |
| section | 0.4821 | 0.3928 | 0.4329 | 14774 |
| dialogue | 0.0275 | 0.1621 | 0.0470 | 2258 |
| list_item | 0.1339 | 0.6269 | 0.2206 | 18815 |
| metadata | 0.6826 | 0.7388 | 0.7096 | 50602 |

Full 5k boundary-head metrics:

| label.task | precision | recall | F1 | validation positives |
| --- | ---: | ---: | ---: | ---: |
| sentence.start | 0.3240 | 0.3472 | 0.3352 | 599 |
| sentence.end | 0.4815 | 0.2664 | 0.3430 | 488 |
| paragraph.start | 0.1832 | 0.0860 | 0.1171 | 279 |
| paragraph.end | 0.0851 | 0.1170 | 0.0985 | 171 |
| metadata.start | 0.2920 | 0.3540 | 0.3200 | 226 |
| metadata.end | 0.3897 | 0.2409 | 0.2978 | 220 |

Interpretation:

- adding the full corrected 5k set materially improves `inside` quality over the 2,963-row snapshot
- sentence and paragraph inside labels are now strong enough to be useful region signals
- section and list-item improve but are still not production-quality
- dialogue remains data-limited; there are only `73` dialogue spans in the full corrected 5k set
- full `inside/start/end` training still does not solve segmentation; boundary heads need a different objective/sampling strategy

## Synthetic Semantic Boundary Burn 5k Slice

Source data:

- `data/synthetic/kl3m_streaming_spans_20260429_per_label_5k.jsonl`

Source reports:

- `/tmp/charstreamer-synthetic-boundary-burn-5k-best-inspect.json`
- `/tmp/charstreamer-synthetic-boundary-burn-5k-shape-quote-mask-mlp2_128_64.json`

Configuration:

- model: Burn CPU `NdArray` MLP, hidden dims `128/64`, `32` epochs, batch `1024`
- features: encoded byte window `7/7`, ASCII classes, reusable `BoundaryShapeAppender`, directional byte-class counts radius `24`, directional Unicode group counts radius `24`, line-structure counts
- candidate contract: sentence candidates are terminal punctuation candidates with trailing quote/bracket absorption; paragraph decoding is masked to paragraph-eligible candidates such as blank-line boundaries, document end, and gold paragraph ends during validation
- environment: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`

| experiment | feature_dim | train_rows | valid_rows | train_s | valid rows/s | macro F1 | sentence F1 | paragraph F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| boundary_mlp_128_64_initial | 71 | 4344 | 2896 | 0.662 | 479921.5 | 0.6486 | 0.7495 | 0.5476 |
| boundary_mlp_128_64_shape_quote_mask | 93 | 4358 | 2901 | 0.736 | 428300.2 | 0.6834 | 0.7676 | 0.5992 |

Best current confusion details:

| output | eligible rows | positives | threshold | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sentence_break | 1473 | 488 | 0.51 | 0.6672 | 0.9037 | 0.7676 |
| paragraph_break | 1839 | 171 | 0.22 | 0.4686 | 0.8304 | 0.5992 |

Inspection output on clean prose:

```text
<|sentence|>The agreement was signed on March 4, 2026.<|/sentence|> <|sentence|>The seller delivered the equipment three days late, but the buyer accepted the shipment without reservation.<|/sentence|> <|sentence|>After reviewing the invoices, the court finds that the delay did not cause measurable damages.<|/sentence|> <|sentence|>The request for penalties is denied.<|/sentence|>
```

Inspection output on mixed legal/news-like text still shows the expected next failure mode: sentence decoding includes metadata and headings in the first sentence span, because the current model only predicts sentence and paragraph breakpoints. Quote handling is fixed relative to the previous run; `"Nobody could locate the missing pallets."` is emitted as one valid sentence span ending after the closing quote instead of breaking before it.

Interpretation:

- boundary-specific training is the right path for segmentation; it is far better than the shared `inside/start/end` heads for actual break prediction
- the reusable `BoundaryShapeAppender` and quote-aware candidate normalization improve both sentence and paragraph quality with only a small feature-width increase
- paragraph quality is now mostly limited by task definition and candidate eligibility, not by raw model architecture
- metadata, heading, list-item, dialogue, and section boundaries need their own output heads or task-specific decoders; sentence/paragraph breakpoints alone cannot correctly segment mixed structural text

### Archived Burn vs SmartCore on the 5k Synthetic Boundary Task

After removing Candle, the synthetic boundary trainer was rerun on the same
5k corrected per-label dataset with the same split seed and feature stack.
SmartCore was then removed from the active dependency path; the Burn command
below is the supported rerun path.

Commands:

```bash
cargo run --release -p charstreamer-experiments --example train_synthetic_boundary_burn -- \
  --input data/synthetic/kl3m_streaming_spans_20260429_per_label_5k.jsonl \
  --report /tmp/charstreamer-synthetic-boundary-burn-5k-rerun-after-candle-drop.json \
  --epochs 32 --batch-size 1024 \
  --hidden-dim 128 --hidden-dim2 64 \
  --encoded-left 7 --encoded-right 7 --count-radius 24 --seed 19
```

| model | train rows | valid rows | train s | valid rows/s | valid e2e chars/s | macro F1 | sentence F1 | paragraph F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Burn MLP `128/64`, 32 epochs | 11,630 | 2,901 | 1.283 | 1,099,505.8 | 12,974,649.2 | 0.7033 | 0.7823 | 0.6242 |
| SmartCore decision tree, depth 12 | 11,630 | 2,901 | 0.617 | 1,062,608.2 | 12,996,178.5 | 0.6509 | 0.7614 | 0.5404 |

Interpretation:

- Burn still works end-to-end after Candle removal and currently wins quality on the synthetic boundary task
- SmartCore decision trees were a useful fast baseline and trained about `2.1x` faster here, but lost `0.0524` macro F1
- paragraph breaks are where the SmartCore tree loses most of the gap

## Synthetic Structure Model and Post-Merge Slice

Source data:

- `data/synthetic/kl3m_streaming_spans_20260429_per_label_5k.jsonl`

Source reports:

- `/tmp/charstreamer-structure-lines-5k-prior-merge.json`

Implementation:

- `crates/charstreamer-experiments/examples/train_synthetic_structure_burn.rs`

Configuration:

- model: Burn CPU `NdArray` line-candidate MLP, hidden dims `128/64`, `32` epochs, batch `512`
- labels: `metadata`, `section`, `list_item`, `dialogue`
- candidates: non-empty logical lines, snapped to non-whitespace line bounds
- features: encoded byte window `7/7`, ASCII classes, reusable `BoundaryShapeAppender`, directional byte-class counts radius `32`, directional Unicode group counts radius `32`, line-structure counts
- merge decoder: line-level semantic spans plus paragraph spans plus sentence spans; metadata/section/list-item spans suppress sentence spans, dialogue can contain nested sentence spans
- decoder priors: obvious structural line forms (`#` headings, bullet/list markers, metadata key/value lines before first blank line, quoted dialogue) are used as decoder priors on top of model scores

Raw line-model validation metrics:

| output | positives | threshold | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| metadata | 441 | 0.20 | 0.4226 | 0.6621 | 0.5159 |
| section | 204 | 0.19 | 0.4342 | 0.4853 | 0.4583 |
| list_item | 82 | 0.08 | 0.1899 | 0.5488 | 0.2821 |
| dialogue | 10 | 0.01 | 0.0000 | 0.0000 | 0.0000 |

Training/runtime:

| train_rows | valid_rows | feature_dim | train_s | valid rows/s | valid e2e chars/s | raw macro F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10015 | 2557 | 81 | 1.840 | 427859.2 | 12336321.9 | 0.3141 |

Post-merged inspection output:

```text
<|paragraph|><|metadata|>Case Note: Synthetic v. Example
Docket: 26-CV-1042
Date: April 28, 2026</|metadata|></|paragraph|>

<|paragraph|><|section|># Background</|section|>
<|sentence|>The court reviewed the contract and the attached invoices.</|sentence|> <|sentence|>The vendor argued that the late delivery was excused by a port closure.</|sentence|></|paragraph|>

<|paragraph|><|section|># Findings</|section|>
<|sentence|>First, the shipment logs show a gap of six days.</|sentence|> <|sentence|>Second, the buyer sent two written notices before canceling the order.</|sentence|></|paragraph|>

<|paragraph|><|list_item|>- The refund request was timely.</|list_item|>
<|list_item|>- The replacement goods were accepted without objection.</|list_item|>
<|list_item|>- Further interest is denied.</|list_item|></|paragraph|>

<|paragraph|><|dialogue|><|sentence|>"I called the warehouse twice," Maria said.</|sentence|> <|sentence|>"Nobody could locate the missing pallets."</|sentence|></|dialogue|></|paragraph|>

<|paragraph|><|sentence|>Conclusion: judgment is entered for the buyer in part.</|sentence|></|paragraph|>
```

Interpretation:

- the right architecture is multi-stage: boundary models for high-frequency sentence/paragraph breaks, and line/span candidate models for structural semantic labels
- the raw line model is useful for metadata and section but not enough for rare labels yet; `dialogue` has only `10` validation positives in this split, so the learned line model alone cannot be trusted
- decoder priors are not a hack; they are the right place to encode deterministic candidate facts like `#` headings and bullet lines while keeping the model responsible for ambiguous cases
- the current merge renderer emits a single nested annotation stream, but the canonical internal representation should remain standoff spans because future labels can overlap in ways that are not always XML-nestable

## Production Combined Segmenter Benchmark

Implementation:

- `crates/charstreamer-segmentation`
- `crates/charstreamer-python`
- `crates/charstreamer-segmentation/benches/long_document.rs`

Benchmark input:

- `data/bench/war_and_peace.txt`
- source URL used locally: `https://www.gutenberg.org/cache/epub/2600/pg2600.txt`
- file size from `wc -c`: `3,359,613` bytes

Command:

```bash
CHARSTREAMER_BENCH_TEXT=/home/mjbommar/projects/personal/charstreamer/data/bench/war_and_peace.txt \
  cargo bench -p charstreamer-segmentation --bench long_document -- \
  --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Criterion result:

| benchmark | mean time | throughput |
| --- | ---: | ---: |
| combined_segmenter_long_document | 93.70 ms | 34.19 MiB/s |

One-shot breakdown from `time_once`:

| stage | time | notes |
| --- | ---: | --- |
| structure spans | 26.65 ms | line/span semantic detection |
| sentence spans | 24.61 ms | punctuation boundary detection |
| full span merge | 77.00 ms | complete standoff span generation |
| tag rendering | 21.40 ms | inline annotation view |
| total | 98.40 ms | `32.56 MiB/s` |

Python release wheel smoke result for the Rust-owned benchmark path:

```text
bytes_per_second=37,125,247.5
chars_per_second=36,381,778.1
mib_per_second=35.41
iterations=3
```

Actual Python `Segmenter.annotate(text)` API result, including conversion of
47,547 spans into Python dictionaries with both character and byte offsets:

```text
mean_s=0.1631
chars_per_second=19,786,842.1
span_count=47,547
```

Interpretation:

- the production default API now runs at roughly `34 MiB/s` end-to-end on a real long UTF-8 document while producing standoff spans and inline tags
- two important production fixes came out of the benchmark: conflict cleanup is now active-window based instead of quadratic, and rendering is event-based instead of doing per-character map lookups
- the Python/PyO3 release path is still fast, but returning span dictionaries has measurable conversion cost; use `Segmenter.benchmark` or a future zero-copy/buffer API to measure pure Rust throughput from Python

## Legacy Window Sweep

Source: `/tmp/charboundary-window-sweep.md`

| experiment | val_f1 | cand_f1 | val_cps | train_s | eval_f1 | eval_cps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| charboundary_small_logistic | 0.7156 | 0.7608 | 22786302.8 | 0.027 | 0.3728 | 9987839.4 |
| charboundary_small_tree | 0.8106 | 0.9220 | 21153603.8 | 0.037 | 0.5106 | 9733144.9 |
| charboundary_medium_logistic | 0.6016 | 0.6770 | 21867562.6 | 0.029 | 0.2570 | 9807063.5 |
| charboundary_medium_tree | 0.8127 | 0.9150 | 20472378.9 | 0.041 | 0.4806 | 9572505.5 |
| charboundary_large_logistic | 0.5595 | 0.6241 | 21169775.9 | 0.031 | 0.1662 | 9629751.7 |
| charboundary_large_tree | 0.8056 | 0.9079 | 19111679.3 | 0.047 | 0.4740 | 9370018.6 |

Interpretation:

- wider legacy windows hurt the logistic path badly on the current reduced sweep
- wider legacy windows did not improve cross-domain tree quality enough to justify becoming the default

## Current Fast Path

Source reports:

- `/tmp/current-legal-logistic-report.json`
- `/tmp/current-legal-tree-report.json`

| experiment | val_f1 | cand_f1 | val_cps | train_s | eval_f1 | eval_cps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_legal_logistic | 0.7106 | 0.6728 | 69362967.7 | 0.020 | 0.4015 | 1187890.7 |
| current_legal_tree | 0.7976 | 0.8450 | 52545948.4 | 0.036 | 0.5142 | 1188319.4 |

Interpretation:

- moving from logistic to tree on the same fast feature family materially improves quality
- throughput remains much higher than the legacy tree/RF baselines

## Fast Tree Feature Ablation

Source reports:

- `/tmp/current-legal-tree-report.json`
- `/tmp/current-legal-tree-counts-report.json`
- `/tmp/current-legal-tree-ablation.md`
- `/tmp/current-legal-tree-local-split.md`

### Count block ablation

| experiment | val_f1 | cand_f1 | val_cps | train_s | eval_f1 | eval_cps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_legal_tree | 0.7976 | 0.8450 | 53240129.0 | 0.047 | 0.5142 | 1186344.1 |
| current_legal_tree_local_counts | 0.8157 | 0.8803 | 39746565.1 | 0.056 | 0.5082 | 1172896.9 |
| current_legal_tree_line_counts | 0.8016 | 0.8517 | 40475987.2 | 0.047 | 0.5090 | 1139964.3 |
| current_legal_tree_counts | 0.8078 | 0.8636 | 32053271.0 | 0.056 | 0.5155 | 1134305.7 |

### Local-count split

| experiment | val_f1 | cand_f1 | val_cps | train_s | eval_f1 | eval_cps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_legal_tree | 0.7976 | 0.8450 | 53977472.5 | 0.047 | 0.5142 | 1187733.4 |
| current_legal_tree_local_punct | 0.8157 | 0.8803 | 45741468.2 | 0.052 | 0.5075 | 1185124.3 |
| current_legal_tree_local_structure | 0.8196 | 0.8880 | 46104332.7 | 0.046 | 0.5222 | 1185352.2 |
| current_legal_tree_local_counts | 0.8157 | 0.8803 | 40486155.4 | 0.056 | 0.5082 | 1181274.6 |

Interpretation:

- local structure counts are the strongest current feature addition on the fast tree path
- line counts do not justify themselves on the current sweep
- combining the local punctuation and structure counts is worse than structure-only

### Class-count ablation

| experiment | val_f1 | cand_f1 | val_cps | train_s | eval_f1 | eval_cps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_legal_tree | 0.7976 | 0.8450 | 53983620.0 | 0.047 | 0.5142 | 1187011.4 |
| current_legal_tree_local_structure | 0.8196 | 0.8880 | 46142085.3 | 0.047 | 0.5222 | 1183735.6 |
| current_legal_tree_class_counts | 0.8297 | 0.9077 | 25404812.3 | 0.048 | 0.4949 | 1163250.5 |
| current_legal_tree_structure_class_counts | 0.8297 | 0.9077 | 23446263.2 | 0.053 | 0.4951 | 1159303.6 |

Interpretation:

- symmetric byte-class density features improve ALEA validation metrics but hurt SCOTUS generalization
- stacking symmetric class counts on top of the existing structure-count block makes that overfitting worse, not better

### Directional class-count ablation

| experiment | val_f1 | cand_f1 | val_cps | train_s | eval_f1 | eval_cps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_legal_tree | 0.7976 | 0.8450 | 53363885.0 | 0.046 | 0.5142 | 1186572.6 |
| current_legal_tree_local_structure | 0.8196 | 0.8880 | 45416629.1 | 0.046 | 0.5222 | 1182888.7 |
| current_legal_tree_directional_class_counts | 0.8126 | 0.8750 | 35021154.8 | 0.056 | 0.5198 | 1175166.8 |
| current_legal_tree_structure_directional_class_counts | 0.8024 | 0.8549 | 31525204.9 | 0.060 | 0.5170 | 1171927.1 |

Interpretation:

- left/right-separated byte-class densities generalize better than symmetric class counts
- they still do not beat the simpler local structure-count block on cross-domain F1
- combining directional class densities with the existing structure-count block is not additive on the current sweep

### Tree tuning on the best feature preset

| experiment | val_f1 | cand_f1 | val_cps | train_s | eval_f1 | eval_cps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_legal_tree_local_structure | 0.8196 | 0.8880 | 46153978.3 | 0.048 | 0.5222 | 1181739.4 |
| current_legal_tree_local_structure_balanced | 0.8040 | 0.8583 | 45805822.6 | 0.041 | 0.5002 | 1183609.9 |
| current_legal_tree_local_structure_shallow | 0.8000 | 0.8525 | 46481980.9 | 0.036 | 0.4946 | 1182960.8 |
| current_legal_tree_local_structure_entropy | 0.8268 | 0.9027 | 45664610.5 | 0.046 | 0.5141 | 1183100.2 |

Interpretation:

- the original `current_legal_tree_local_structure` hyperparameters still generalize best on the current reduced SCOTUS sweep
- Entropy helps in-split validation but not cross-domain quality
- shallow regularization cuts training time a little but loses too much quality to become the default

## Fast Logistic Feature Check

Source sweep:

- `current_legal_logistic`
- `current_legal_logistic_local_structure`

| experiment | val_f1 | cand_f1 | val_cps | train_s | eval_f1 | eval_cps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_legal_logistic | 0.7106 | 0.6728 | 68358521.7 | 0.020 | 0.4015 | 1192688.1 |
| current_legal_logistic_local_structure | 0.6096 | 0.5263 | 56611727.0 | 0.022 | 0.3752 | 1190349.8 |

Interpretation:

- the local structure count block is tree-friendly, not logistic-friendly
- keep it on the tree preset, not on the logistic preset

## Qualitative Boundary Comparison on Fiction

Input:

- source: War and Peace from `data/bench/war_and_peace.txt`
- sample policy: deterministic random windows, seed `20260429`
- sample count: 20
- window size: roughly 500-1200 Python characters
- models: `charstreamer` release wheel, `nupunkt` default model, original `charboundary` small/medium/large ONNX models

This is not a gold-label evaluation. Agreement with `charboundary-medium-onnx`
is only a proxy for model disagreement and qualitative inspection.

| model | mean time ms | median time ms | mean sentence boundaries | exact Jaccard vs cb-medium | near agreement vs cb-medium, tol 2 chars |
| --- | ---: | ---: | ---: | ---: | ---: |
| `charstreamer` | 0.187 | 0.185 | 6.60 | 0.837 | 0.853 |
| `nupunkt` | 0.344 | 0.299 | 6.20 | 0.844 | 0.860 |
| `charboundary-small-onnx` | 2.424 | 2.344 | 7.40 | 0.953 | 0.953 |
| `charboundary-medium-onnx` | 2.664 | 2.691 | 7.10 | 1.000 | 1.000 |
| `charboundary-large-onnx` | 3.334 | 2.967 | 7.20 | 0.976 | 0.976 |

Representative findings:

- simple prose is easy: all models agreed exactly on a 624-character prose sample with 4 sentence boundaries
- `charstreamer` and `nupunkt` are close on average, with `charstreamer` faster on these short windows
- original `charboundary` ONNX models split more aggressively in dialogue, parentheticals, and semicolon-heavy passages
- `charstreamer` currently under-splits some dialogue fragments, for example keeping `"You're here?" he seemed...` together while `charboundary` splits after the quoted question
- `charboundary` sometimes splits quote attribution and semicolon clauses more aggressively than a conservative sentence tokenizer would; this can be desirable or excessive depending on the target annotation policy
- the Python binding now reports `start`/`end` as Python character offsets and keeps canonical Rust offsets as `start_byte`/`end_byte`; an earlier ad hoc comparison incorrectly treated byte offsets as character offsets on Unicode text

## Current Default Candidates

- strongest full-corpus single-thread default candidate: `current_legal_tree_directional_class_counts_window_3_1_full`
- strongest full-corpus quality-oriented tree candidate: `current_legal_tree_directional_class_counts_window_3_3_full`
- strongest reduced-sweep default candidate: `current_legal_tree_local_structure`
- strongest pure-speed preset candidate: `current_legal_logistic`
- strongest legacy-parity docking baseline: `charboundary_small_tree`
- strongest Burn neural baseline: `current_legal_burn_window_cnn_directional_full`
- strongest current semantic segmentation boundary slice: `boundary_mlp_128_64_shape_quote_mask`
- best current generic class-density primitive: `DirectionalByteClassCounts`; it is now part of the best full-corpus default preset
- best current Unicode-aware primitive: `DirectionalUnicodeCategoryGroupCounts`; it is correct and configurable, but it is not yet part of the default preset because it regresses SCOTUS F1 on the current corpora
