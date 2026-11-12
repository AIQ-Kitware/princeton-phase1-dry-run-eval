"""CLI surface for the CCG v2 law.

One verb:

  ``score``  -- read ladder files, fit each model on its own calibration rungs (default
                T = 0, 1, 2), predict the held-out rungs (default T = 3, 4), bootstrap over problems,
                and write a summary JSON. Any ladder file in the format of
                ``data/ccg_v2/ladders`` is accepted::

    python -m contextual_drag analysis ccg_v2 score \\
        --ladders data/ccg_v2/ladders/*.json --out scores.json
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import scriptconfig as scfg


def _rungs(text):
    """
    Example:
        >>> _rungs('0,1,2')
        (0, 1, 2)
    """
    if isinstance(text, (list, tuple)):
        return tuple(int(t) for t in text)
    return tuple(int(t) for t in str(text).replace(' ', '').split(',') if t)


class CcgV2ScoreCLI(scfg.DataConfig):
    """Fit the CCG v2 law on each model's calibration rungs and score its predictions."""
    ladders = scfg.Value(None, required=True, nargs='+', help='Ladder JSON files, one per model.',
                         tags=['in_path'])
    calibration_rungs = scfg.Value('0,1,2', type=str, tags=['algo_param'])
    predict_rungs = scfg.Value('3,4', type=str, tags=['algo_param'])
    tolerance_pp = scfg.Value(5.0, type=float, tags=['algo_param'])
    n_boot = scfg.Value(2000, type=int, help='0 skips the bootstrap.', tags=['algo_param'])
    seed = scfg.Value(20261111, type=int, tags=['algo_param'])
    confidence = scfg.Value(0.95, type=float, tags=['algo_param'])
    out = scfg.Value(None, required=True, tags=['out_path'])

    @classmethod
    def main(cls, argv=None, **kwargs):
        cfg = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        result = run_from_files(cfg.ladders, calibration_rungs=_rungs(cfg.calibration_rungs),
                                predict_rungs=_rungs(cfg.predict_rungs), tolerance_pp=float(cfg.tolerance_pp),
                                n_boot=int(cfg.n_boot), seed=int(cfg.seed), confidence=float(cfg.confidence))
        out = Path(cfg.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        for name, m in result['per_model'].items():
            errs = ', '.join(f"T{T} {m['err_pp'][str(T)]:+.2f}" for T in result['config']['predict_rungs'])
            print(f"[ccg_v2] {name}: c={m['c']:.2f} w={m['w']:.2f} {errs} pass={m['pass']}")


def _seed_part(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def score_payload(payload, *, calibration_rungs=(0, 1, 2), predict_rungs=(3, 4), tolerance_pp=5.0,
                  n_boot=2000, seed=20261111, confidence=0.95):
    """Score one ladder. Errors are predicted minus measured accuracy, in percentage points."""
    import numpy as np
    from contextual_drag.analysis.ccg_v2.model import fit, line_baseline, predict, rung_accuracy
    K = int(payload.get('K') or 4)
    acc, n = rung_accuracy(payload)
    c, w = fit(acc, calibration_rungs, n, K)
    out = dict(model=str(payload['model']), n_problems=int(n), rung_obs={str(T): acc[T] for T in sorted(acc)},
               c=c, w=w, tie_accuracy=1 / (1 + math.exp(-c)),
               calibration_resid_pp={str(T): 100 * (predict(c, w, T, K) - acc[T]) for T in calibration_rungs},
               pred={str(T): predict(c, w, T, K) for T in predict_rungs},
               err_pp={str(T): 100 * (predict(c, w, T, K) - acc[T]) for T in predict_rungs},
               line_err_pp={str(T): 100 * (line_baseline(acc, calibration_rungs, T) - acc[T]) for T in predict_rungs})
    out['pass'] = all(abs(e) <= tolerance_pp for e in out['err_pp'].values())
    out['line_pass'] = all(abs(e) <= tolerance_pp for e in out['line_err_pp'].values())
    if n_boot > 0:
        rng = np.random.default_rng([seed, _seed_part(out['model'])])
        ids = [str(p['id']) for p in payload['problems']]
        draws = {f'err_t{T}': [] for T in predict_rungs}
        draws.update(c=[], w=[])
        for _ in range(n_boot):
            wts = {}
            for i in rng.integers(0, len(ids), len(ids)):
                wts[ids[i]] = wts.get(ids[i], 0.0) + 1.0
            a, nb = rung_accuracy(payload, wts)
            cb, wb = fit(a, calibration_rungs, nb, K)
            for T in predict_rungs:
                draws[f'err_t{T}'].append(100 * (predict(cb, wb, T, K) - a[T]))
            draws['c'].append(cb)
            draws['w'].append(wb)
        alpha = 1 - confidence
        out['ci'] = {k: [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))]
                     for k, v in draws.items()}
        out['supported'] = all(-tolerance_pp <= out['ci'][f'err_t{T}'][0] and out['ci'][f'err_t{T}'][1] <= tolerance_pp
                               for T in predict_rungs)
    return out


def run_from_files(ladder_fpaths, *, calibration_rungs=(0, 1, 2), predict_rungs=(3, 4), tolerance_pp=5.0,
                   n_boot=2000, seed=20261111, confidence=0.95):
    """Score every ladder and summarize."""
    payloads = [json.loads(Path(f).read_text()) for f in ladder_fpaths]
    names = [str(p['model']) for p in payloads]
    if len(set(names)) != len(names):
        raise ValueError(f'duplicate models among ladders: {names}')
    per_model = {str(p['model']): score_payload(p, calibration_rungs=calibration_rungs, predict_rungs=predict_rungs,
                                                tolerance_pp=tolerance_pp, n_boot=n_boot, seed=seed,
                                                confidence=confidence)
                 for p in sorted(payloads, key=lambda p: str(p['model']))}
    errs = [abs(e) for m in per_model.values() for e in m['err_pp'].values()]
    line = [abs(e) for m in per_model.values() for e in m['line_err_pp'].values()]
    return dict(
        n_models=len(per_model),
        n_models_pass=sum(m['pass'] for m in per_model.values()),
        n_models_supported=sum(bool(m.get('supported')) for m in per_model.values()),
        n_models_line_pass=sum(m['line_pass'] for m in per_model.values()),
        mae_pp=sum(errs) / len(errs) if errs else None,
        max_abs_err_pp=max(errs) if errs else None,
        line_mae_pp=sum(line) / len(line) if line else None,
        tolerance_pp=tolerance_pp,
        config=dict(calibration_rungs=list(calibration_rungs), predict_rungs=list(predict_rungs), n_boot=n_boot,
                    seed=seed, confidence=confidence, ladders=[str(f) for f in ladder_fpaths],
                    ladder_sha256={str(f): hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ladder_fpaths}),
        per_model=per_model)


class CcgV2CLI(scfg.ModalCLI):
    score = CcgV2ScoreCLI
