# Contextual Drag: CCG v2

**Princeton TA1 (Princeton-2) — Phase I evaluation card for DARPA AIQ, version 2**

This branch contains `cards/ccg_v2.yaml`. The card tests whether a two-parameter law of contextual drag, fitted on a model's accuracy with zero, one and two correct drafts in its context, predicts its accuracy with three and four. **It generates nothing**: its only stage scores ladder files that already exist, so it runs in seconds on a CPU, with no GPU and no model endpoint. It runs on [AIQ-MAGNET](https://github.com/AIQ-Kitware/aiq-magnet) 0.1.0.

Version 1, the route-model card filed as AIQ-Kitware/aiq-evaluations#44, is documented in [README_CCG_TRANSFER_V1.md](README_CCG_TRANSFER_V1.md); the dry-run cards in [README_LEGACY.md](README_LEGACY.md).

## 1. The Law

With $`K = 4`$ drafts of which $`T`$ are correct, model $`m`$ answers correctly with probability

```math
\operatorname{logit} p_m(T) = c_m + w_m\,\ell(T), \qquad \ell(T) = \log\frac{T + 1/2}{K - T + 1/2}.
```

Equivalently, the model's odds of answering correctly are $`e^{c_m}`$ times the **draft odds** $`(T + 1/2)/(K - T + 1/2)`$ raised to the power $`w_m`$; the ½ is the Jeffreys pseudo-count [2].
- $`c_m`$ is the log-odds of a correct answer when the drafts are tied two against two ($`\ell(2) = 0`$).
- $`w_m`$ is the model's sensitivity to the draft odds: at 1 its odds follow them in proportion; above 1 it amplifies a majority; below 1 it relies more on its own judgment.
- $`\ell(K - T) = -\ell(T)`$: in log-odds the upper half of a ladder mirrors the lower half, so the lower half predicts the upper.

## 2. The Test

| Setting | Value |
|---|---|
| Data | Six competition-mathematics ladders (AIME and HMMT 2024–25) in `data/ccg_v2/ladders`: GPT-OSS-120B, GPT-OSS-20B, Gemma4-31B, Gemma4-E4B, Qwen3-32B, Qwen3-8B; 18–30 eligible problems each; provenance and sha256 in `PROVENANCE.md` |
| Fit | $`(c_m, w_m)`$ per model by weighted least squares on the log-odds of its accuracies at $`T = 0, 1, 2`$; nothing from another model, no tuned constant |
| Prediction | Accuracy at $`T = 3`$ and $`T = 4`$ |
| Pass criterion | Every model's two errors at most 5.0 percentage points |
| Reported | 95% bootstrap intervals over problems; the straight-line extrapolation through $`T = 1, 2`$ as a no-law baseline |

## 3. Result on the Bundled Ladders

| Model | $`c_m`$ | $`w_m`$ | Error at $`T = 3`$ | Error at $`T = 4`$ |
|---|---|---|---|---|
| GPT-OSS-120B | 1.76 | 1.27 | −3.1 | −0.2 |
| GPT-OSS-20B | 1.80 | 0.93 | −3.6 | −0.4 |
| Gemma4-31B | 2.09 | 1.06 | +0.2 | +0.7 |
| Gemma4-E4B | 1.75 | 1.57 | +0.7 | +1.2 |
| Qwen3-32B | 1.24 | 1.14 | −0.2 | −1.7 |
| Qwen3-8B | 1.67 | 1.14 | −3.2 | −0.8 |

All six pass, so MAGNET reports VERIFIED (mean absolute error 1.3 points; the straight line: 2.4, five of six). **Caveats:** the law and the calibration rungs were chosen after examining these ladders, so this is a reproduction, not a blind test; and the predicted rungs are near the ceiling, where prediction is forgiving. Ladders of models not used to design the law are the real test, and the card scores any such ladder unchanged (Section 4.2).

## 4. Running

### 4.1 The bundled ladders

```bash
git checkout phase1-final/ccg-transfer-v2
pip install -e ".[magnet,analysis]"          # no inference extras needed
bash scripts/run_ccg_v2.sh          # about 15 s on a CPU
```

The log prints the per-model errors on lines beginning `[ccg_v2]`; the full result is `_kwdagger/ccg_v2/<id>/results.json` and `ccg_v2_scores.json` under `OUTPUT_PATH` (default `evaluation_runs`).

### 4.2 Ladders from a version-1 run

Every ladder file written by the earlier cards (`<OUTPUT_PATH>/_kwdagger/ladder/<id>/ladder.json`) is scored as it is. In the card's matrix set

```yaml
ccg_v2.ladders_dpath: '<OUTPUT_PATH>/_kwdagger/ladder'
ccg_v2.ladders_glob: '*/ladder.json'
ccg_v2.n_targets_expected: <models in that run>
ccg_v2.min_models_pass: <models in that run>
```

and set the claim symbols `n_targets_expected` and `min_models_pass` to the same number. Without MAGNET:

```bash
PYTHONPATH=$PWD:$PWD/src python -m contextual_drag analysis ccg_v2 score \
    --ladders <OUTPUT_PATH>/_kwdagger/ladder/*/ladder.json --out ccg_v2_scores.json
```

## 5. Components

```
cards/ccg_v2.yaml   evaluation card: claim, settings, theory links
cards/pipelines.py                           ccg_v2_pipeline(): one stage, no generation
cards/nodes/cd_ccg_v2.py                   the stage: find ladders, score, emit the verdict
src/contextual_drag/analysis/ccg_v2/       the law, the fit, the bootstrap, the CLI
data/ccg_v2/ladders/                         the six bundled ladders and their provenance
theory/indexes/ccg_v2.yaml
scripts/run_ccg_v2.sh
tests/test_ccg_v2.py                       reproduction of the note, node behaviour, no generation
```

## 6. Testing

```bash
PYTHONPATH=$PWD:$PWD/src pytest tests/test_ccg_v2.py -q
```

The tests check that the bundled ladders reproduce the theory note's parameters and errors, that the ladders match their recorded sha256, that the stage scores a version-1 store layout and is INCONCLUSIVE without ladders, and that the pipeline has no generation stage.

## References

[1] Yun Cheng, Xingyu Zhu, Haoyu Zhao and Sanjeev Arora. *Contextual Drag: How Errors in the Context Affect LLM Reasoning.* arXiv:2602.04288, 2026.

[2] Team Princeton-2. *Contextual Drag as a Log-Count Vote.* Theory note, November 11, 2026.

## Contact

Yun Cheng (yc6206@princeton.edu)
