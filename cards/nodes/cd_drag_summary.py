"""
Terminal node: compute contextual drag and emit the card's whole result.

Fans in three upstream artifacts -- the processed clean-round dataset, the
2F dataset, and the 2F error analysis -- because the headline number is a
difference between two accuracies measured on the *same* set of problems.

``acc_clean`` is deliberately restricted to the problems that survived the
aggregate filter. Comparing an unrestricted clean accuracy against a 2F
accuracy measured only on survivors would attribute the filter's selection
effect to contextual drag.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import scriptconfig as scfg

from cards.nodes._step import read_manifest, write_manifest

#: Bump when the terminal-result shape changes incompatibly.
SCHEMA_VERSION = 1


class CDDragSummaryCLI(scfg.DataConfig):
    """Compute acc_clean, acc_2f and their difference."""

    aggregate_manifest_fpath = scfg.Value(
        None, help='Manifest from the aggregate node (2F dataset + kept ids).',
        tags=['in_path'])
    twof_eval_manifest_fpath = scfg.Value(
        None, help='Manifest from the 2F eval node (error analysis).',
        tags=['in_path'])

    drag_threshold = scfg.Value(
        0.05, type=float,
        help='Claim threshold: drag must exceed this to verify.',
        tags=['algo_param'])
    min_kept_problems = scfg.Value(
        1, type=int,
        help='Below this many surviving problems the result is not '
             'interpretable and the card reports INCONCLUSIVE.',
        tags=['algo_param'])

    summary_fpath = scfg.Value(
        'results.json', help="The card's terminal artifact.",
        tags=['out_path', 'primary'])

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        summary_fpath = Path(config.summary_fpath).resolve()
        aggregate = read_manifest(config.aggregate_manifest_fpath)
        twof_eval = read_manifest(config.twof_eval_manifest_fpath)

        if aggregate.get('skipped'):
            _emit(summary_fpath, config, status='INCONCLUSIVE',
                  detail=aggregate.get('reason', 'no 2F dataset'),
                  acc_clean=None, acc_2f=None, drag=None, n_kept_problems=0)
            return

        acc_clean = _restricted_clean_accuracy(
            Path(aggregate['processed_ds']), Path(aggregate['dataset_fpath']))
        acc_2f = _twof_accuracy(twof_eval.get('error_analysis_fpath'))
        n_kept = int(aggregate.get('n_kept') or 0)

        if acc_clean is None or acc_2f is None:
            _emit(summary_fpath, config, status='INCONCLUSIVE',
                  detail='one of the two accuracies could not be computed',
                  acc_clean=acc_clean, acc_2f=acc_2f, drag=None,
                  n_kept_problems=n_kept)
            return

        if n_kept < int(config.min_kept_problems):
            _emit(summary_fpath, config, status='INCONCLUSIVE',
                  detail=(f'only {n_kept} problems survived the filter; '
                          f'need >= {config.min_kept_problems}'),
                  acc_clean=acc_clean, acc_2f=acc_2f,
                  drag=acc_clean - acc_2f, n_kept_problems=n_kept)
            return

        drag = acc_clean - acc_2f
        verified = drag > float(config.drag_threshold)
        _emit(summary_fpath, config,
              status='VERIFIED' if verified else 'FALSIFIED',
              detail='' if verified else (
                  f'drag {drag:.4f} <= threshold {config.drag_threshold}'),
              acc_clean=acc_clean, acc_2f=acc_2f, drag=drag,
              n_kept_problems=n_kept)


def _restricted_clean_accuracy(processed_ds: Path, twof_ds: Path):
    """Clean accuracy over only the problems that survived the filter."""
    from datasets import load_from_disk

    clean = load_from_disk(str(processed_ds))
    kept_ids = set(load_from_disk(str(twof_ds))['id'])

    per_problem = defaultdict(lambda: {'correct': 0, 'total': 0})
    for entry in clean:
        if entry['id'] not in kept_ids:
            continue
        per_problem[entry['id']]['total'] += 1
        if entry['init_response_generations_correctness']:
            per_problem[entry['id']]['correct'] += 1

    total = sum(p['total'] for p in per_problem.values())
    if not total:
        return None
    return sum(p['correct'] for p in per_problem.values()) / total


def _twof_accuracy(error_analysis_fpath):
    """Overall 2F correctness from the evaluator's error analysis."""
    if not error_analysis_fpath:
        return None
    with open(error_analysis_fpath, 'r') as file:
        analysis = json.load(file)
    try:
        return float(
            analysis['pass_at_k_by_source']['overall']['overall_correctness'])
    except (KeyError, TypeError, ValueError):
        return None


def _emit(summary_fpath, config, *, status, detail, acc_clean, acc_2f, drag,
          n_kept_problems):
    write_manifest(
        summary_fpath,
        schema_version=SCHEMA_VERSION,
        status=status,
        detail=detail,
        metrics={'acc_clean': acc_clean, 'acc_2f': acc_2f, 'drag': drag,
                 'drag_threshold': float(config.drag_threshold)},
        cohort={'n_kept_problems': n_kept_problems},
    )
    drag_str = 'n/a' if drag is None else f'{drag:+.4f}'
    clean_str = 'n/a' if acc_clean is None else f'{acc_clean:.4f}'
    twof_str = 'n/a' if acc_2f is None else f'{acc_2f:.4f}'
    print(f'[drag_summary] status={status} acc_clean={clean_str} '
          f'acc_2f={twof_str} drag={drag_str} n_kept={n_kept_problems}',
          flush=True)
    if detail:
        print(f'  {detail}', flush=True)


__cli__ = CDDragSummaryCLI

if __name__ == '__main__':
    CDDragSummaryCLI.main()
