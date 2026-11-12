# Contextual Drag in Competition Mathematics: CCG v2

**Princeton TA1 (Princeton-2) — Phase I evaluation card for DARPA AIQ**

This repository contains the evaluation card `cards/ccg_v2.yaml`. A model answers a problem with $`K = 4`$ of its own earlier drafts in its context, $`T`$ of them correct. The card tests whether a two-parameter law, fitted to GPT-OSS-20B's accuracy at $`T = 0, 1, 2`$, predicts its accuracy at $`T = 3`$ and $`T = 4`$. The measurements are included in the repository, so the card runs in under a minute on a CPU, without a GPU or model endpoint. It runs on [AIQ-MAGNET](https://github.com/AIQ-Kitware/aiq-magnet) 0.1.0.

## 1. Model

```math
\log\frac{p(T)}{1 - p(T)} = c + w \log\frac{T + 1/2}{K - T + 1/2}
```

$`c`$ is the model's log-odds of a correct answer when its drafts are tied two against two. $`w`$ is how strongly the model follows its drafts.

## 2. Evaluation

| Setting | Value |
|---|---|
| Measurements | GPT-OSS-20B on AIME and HMMT 2024–25. Problems are kept if at least 4 of 16 draft-free answers are correct and at least 4 wrong (22 problems); each is then answered with $`T = 0, \dots, 4`$ correct drafts of the model's own, in eight orders (`data/ccg_v2/ladders/GPT_OSS_20B.json`; protocol in `PROVENANCE.md`) |
| Fit | $`c`$ and $`w`$, by weighted least squares on the log-odds of the accuracies at $`T = 0, 1, 2`$ |
| Prediction | Accuracy at $`T = 3`$ and $`T = 4`$ |
| Pass criterion | Both errors at most 5.0 percentage points |

| MAGNET result | Condition |
|---|---|
| VERIFIED | Both errors are at most 5.0 points. |
| FALSIFIED | An error exceeds 5.0 points. |
| INCONCLUSIVE | The measurements are missing or incomplete. |

## 3. Result

| Model | $`c`$ | $`w`$ | Error at $`T = 3`$ | Error at $`T = 4`$ |
|---|---|---|---|---|
| GPT-OSS-20B | 1.80 | 0.93 | −3.6 | −0.4 |

Errors are predicted minus measured accuracy, in percentage points. Both are within 5.0 points, so MAGNET reports VERIFIED.

**Other runs.** We also evaluate the predictive law on an expanded set of models on competition mathematics (AIME and HMMT 2024–25), measured in the same way (`data/ccg_v2/ladders`).

| Model | $`c`$ | $`w`$ | Error at $`T = 3`$ | Error at $`T = 4`$ |
|---|---|---|---|---|
| GPT-OSS-120B | 1.76 | 1.27 | −3.1 | −0.2 |
| GPT-OSS-20B | 1.80 | 0.93 | −3.6 | −0.4 |
| Gemma4-31B | 2.09 | 1.06 | +0.2 | +0.7 |
| Gemma4-E4B | 1.75 | 1.57 | +0.7 | +1.2 |
| Qwen3-32B | 1.24 | 1.14 | −0.2 | −1.7 |
| Qwen3-8B | 1.67 | 1.14 | −3.2 | −0.8 |

All twelve predictions are within 5.0 points.

## 4. Running

```bash
git clone https://github.com/Princeton-TA1-Team/phase1-dry-run-eval.git
cd phase1-dry-run-eval
git checkout phase1-final/ccg-transfer-v2
pip install -e ".[magnet,analysis]"
bash scripts/run_ccg_v2.sh
```

The verdict is written to `evaluation_runs/<hash>_<time>/verdict.json`, and the per-model errors, parameters and intervals to `evaluation_runs/_kwdagger/ccg_v2/<id>/results.json`.

The results for the expanded set are reproduced with

```bash
PYTHONPATH=$PWD:$PWD/src python -m contextual_drag analysis ccg_v2 score \
    --ladders data/ccg_v2/ladders/*.json --out ccg_v2_scores.json
```

To score other measurements in the same format with the card, set `ccg_v2.ladders_dpath` (an absolute path), `ccg_v2.ladders_glob`, `ccg_v2.n_targets_expected` and `ccg_v2.min_models_pass` in `cards/ccg_v2.yaml`, set the claim symbols `n_targets_expected` and `min_models_pass` to the same number of models, and use a new output path.

## 5. Testing

```bash
pip install -e ".[magnet,analysis,dev]"
PYTHONPATH=$PWD:$PWD/src pytest tests/test_ccg_v2.py tests/test_cards.py -q
```

## Reference

Yun Cheng, Xingyu Zhu, Haoyu Zhao and Sanjeev Arora. *Contextual Drag: How Errors in the Context Affect LLM Reasoning.* arXiv:2602.04288, 2026.

## Contact

Yun Cheng (yc6206@princeton.edu)
