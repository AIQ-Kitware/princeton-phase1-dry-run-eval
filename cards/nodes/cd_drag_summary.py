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
import magnet.theory as theory

#: Bump when the terminal-result shape changes incompatibly.
#: 2 -- cohort gained n_filtered_problems / n_twof_missing, and
#:      n_kept_problems now counts the problems BOTH accuracies were measured
#:      on rather than the problems the aggregate filter selected.
SCHEMA_VERSION = 2


# `drag_threshold` is a card parameter with a default, so the threshold is
# fixed before any result is seen -- which is exactly what `hprespecified`
# asks. Overriding it from the matrix keeps that true; reading a run and then
# picking a threshold would not.
@theory.satisfies('Hygiene.Inference.threshold_exceeds_sampling_error::hprespecified',
                  note='drag_threshold is a card default, fixed before results are seen')
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
    max_twof_loss_frac = scfg.Value(
        0.05, type=float,
        help=('Largest fraction of filtered problems the 2F round may fail to '
              'produce before the result is INCONCLUSIVE. Problems lost to '
              'timeouts or rejected requests are not lost at random -- the '
              'survivors are the ones that answered fastest -- so a drag '
              'measured across the gap is biased, not merely noisier.'),
        tags=['algo_param'])

    summary_fpath = scfg.Value(
        'results.json', help="The card's terminal artifact.",
        tags=['out_path', 'primary'])

    # The card's verdict is `drag > drag_threshold`, so this node is where the
    # hygiene statements are either discharged or left standing.
    #
    # `hn_sufficient` is the load-bearing gap and is deliberately recorded as
    # `assumes` rather than quietly satisfied: `min_kept_problems` is a floor on
    # whether the pipeline produced anything, NOT a check that n resolves a drag
    # of this size. A paired Hoeffding bound at eps = delta = 0.05 needs n >= 738
    # kept problems; runs to date are far below that. The 2026-08-06 scale-up
    # journal records a model reported VERIFIED whose drag was not distinguishable
    # from zero -- found by a person who went looking, which is the failure this
    # annotation exists to make visible without one.
    @theory.tests('Hygiene.Inference.threshold_exceeds_sampling_error')
    @theory.assumes('Hygiene.Inference.threshold_exceeds_sampling_error::hn_sufficient',
                    note='min_kept_problems gates pipeline output, not statistical power; '
                         'n >= 738 is needed at eps = delta = 0.05 and runs are far below')
    @theory.assumes('Hygiene.Inference.threshold_exceeds_sampling_error::hmultiple',
                    note='the model/config sweep is a comparison family; no correction is applied')
    # What the measurement is OF. The construct is "injecting a model's own
    # failed reasoning degrades later accuracy"; the measurement is a
    # benchmark accuracy difference. That gap is worth naming even though
    # nothing here can close it.
    @theory.approximates('Hygiene.Measurement.measured_score_tracks_construct')
    @theory.assumes('Hygiene.Measurement.measured_score_tracks_construct::hcontam',
                    note='no contamination check; GPQA is public and predates the models under test')
    @theory.approximates('Hygiene.Measurement.measured_score_tracks_construct::hstable',
                         note='partly established: the cross-host control in the 2026-08-06 journal '
                              'found Turing fp16 and Ampere bf16 agreeing on acc_clean to 0.0022. '
                              'Hardware and dtype are covered; decoding seed and prompt ordering are not')
    @theory.approximates('Hygiene.Concentration.paired_difference_within_tolerance')
    @theory.assumes('Hygiene.Concentration.paired_difference_within_tolerance::hiid',
                    note='GPQA problems are treated as iid draws; not established')
    @theory.satisfies('Hygiene.Concentration.paired_difference_within_tolerance::hbdd',
                      note='both accuracies are trajectory-weighted means in [0, 1]')
    @theory.assumes('Hygiene.Concentration.paired_difference_within_tolerance::hn',
                    note='same sample-size gap as hn_sufficient')
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

        clean_by_id = _clean_accuracy_by_problem(
            Path(aggregate['processed_ds']), Path(aggregate['dataset_fpath']))
        twof_by_id = _twof_accuracy_by_problem(twof_eval.get('dataset_dir'))
        n_filtered = int(aggregate.get('n_kept') or 0)

        if clean_by_id is None or twof_by_id is None:
            _emit(summary_fpath, config, status='INCONCLUSIVE',
                  detail='one of the two accuracies could not be computed',
                  acc_clean=None, acc_2f=None, drag=None,
                  n_kept_problems=0, n_filtered_problems=n_filtered,
                  n_twof_missing=n_filtered)
            return

        # The paired set, which is the only set on which a difference means
        # anything. Both accuracies are computed over it.
        paired = sorted(set(clean_by_id) & set(twof_by_id))
        n_kept = len(paired)
        n_missing = n_filtered - n_kept

        if not paired:
            _emit(summary_fpath, config, status='INCONCLUSIVE',
                  detail='no problem has both a clean and a 2F accuracy',
                  acc_clean=None, acc_2f=None, drag=None,
                  n_kept_problems=0, n_filtered_problems=n_filtered,
                  n_twof_missing=n_missing)
            return

        acc_clean = _mean_over(clean_by_id, paired)
        acc_2f = _mean_over(twof_by_id, paired)

        if n_kept < int(config.min_kept_problems):
            _emit(summary_fpath, config, status='INCONCLUSIVE',
                  detail=(f'only {n_kept} problems survived the filter; '
                          f'need >= {config.min_kept_problems}'),
                  acc_clean=acc_clean, acc_2f=acc_2f,
                  drag=acc_clean - acc_2f, n_kept_problems=n_kept,
                  n_filtered_problems=n_filtered, n_twof_missing=n_missing)
            return

        # A 2F round that dropped problems did not drop them at random: the
        # survivors are the ones that answered fastest and shortest. Comparing
        # a clean accuracy over the filtered set against a 2F accuracy over
        # that biased remainder is the exact error this node exists to avoid,
        # and it is invisible in the output -- one yardrat run reported
        # n_kept=56 for a drag whose 2F half rested on 12 problems.
        loss_frac = (n_missing / n_filtered) if n_filtered else 0.0
        if loss_frac > float(config.max_twof_loss_frac):
            _emit(summary_fpath, config, status='INCONCLUSIVE',
                  detail=(f'the 2F round produced no output for {n_missing} of '
                          f'{n_filtered} filtered problems ({loss_frac:.0%}); '
                          f'the survivors are not a random subset, so the '
                          f'paired drag over {n_kept} problems is biased. '
                          f'Check the 2F round for timeouts or rejected '
                          f'requests.'),
                  acc_clean=acc_clean, acc_2f=acc_2f,
                  drag=acc_clean - acc_2f, n_kept_problems=n_kept,
                  n_filtered_problems=n_filtered, n_twof_missing=n_missing)
            return

        drag = acc_clean - acc_2f
        verified = drag > float(config.drag_threshold)
        _emit(summary_fpath, config,
              status='VERIFIED' if verified else 'FALSIFIED',
              detail='' if verified else (
                  f'drag {drag:.4f} <= threshold {config.drag_threshold}'),
              acc_clean=acc_clean, acc_2f=acc_2f, drag=drag,
              n_kept_problems=n_kept, n_filtered_problems=n_filtered,
              n_twof_missing=n_missing)


# Both accuracies are computed by this one function over the SAME `ids`, which
# is what makes the drag a per-instance paired difference rather than a
# difference of two independently-sampled means.
@theory.satisfies('Hygiene.Concentration.paired_difference_within_tolerance::hpaired',
                  note='acc_clean and acc_2f are both means over the same surviving problem ids')
def _mean_over(by_id, ids):
    """
    Trajectory-weighted accuracy over a chosen set of problems.

    Args:
        by_id (dict): problem id -> {'correct': int, 'total': int}.
        ids (list): the problems to include.

    Returns:
        float | None

    Example:
        >>> _mean_over({'a': {'correct': 1, 'total': 2}}, ['a'])
        0.5
    """
    total = sum(by_id[i]['total'] for i in ids)
    if not total:
        return None
    return sum(by_id[i]['correct'] for i in ids) / total


# GPQA is multiple choice and scored by exact match against the answer key, so
# the automatic scorer and the construct it proxies coincide here -- which is
# not true of every benchmark this card could be pointed at.
@theory.satisfies('Hygiene.Measurement.measured_score_tracks_construct::hscorer',
                  note='GPQA is multiple choice, scored by exact match against the key')
def _clean_accuracy_by_problem(processed_ds: Path, twof_ds: Path):
    """
    Per-problem clean correctness, restricted to problems that survived.

    Returns:
        dict | None: problem id -> {'correct': int, 'total': int}
    """
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
    return dict(per_problem) or None


def _twof_accuracy_by_problem(dataset_dir):
    """
    Per-problem 2F correctness, read from the evaluated dataset.

    Returns:
        dict | None: problem id -> {'correct': int, 'total': int}

    Deliberately NOT the evaluator's ``overall_correctness`` summary. That is
    an average over whatever the 2F round managed to generate, which is not the
    set the clean accuracy is measured on whenever a request timed out or was
    rejected -- and the resulting drag silently compares two different cohorts.
    """
    if not dataset_dir:
        return None
    path = Path(dataset_dir) / 'evaluated_inference.jsonl'
    if not path.exists():
        return None

    per_problem = defaultdict(lambda: {'correct': 0, 'total': 0})
    with open(path, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            entry = json.loads(line)
            pid = entry.get('id')
            if pid is None:
                continue
            for generation in entry.get('twof_generations') or []:
                if not isinstance(generation, dict):
                    continue
                per_problem[pid]['total'] += 1
                if generation.get('correctness') in (True, 'correct'):
                    per_problem[pid]['correct'] += 1
    return {k: v for k, v in per_problem.items() if v['total']} or None


def _emit(summary_fpath, config, *, status, detail, acc_clean, acc_2f, drag,
          n_kept_problems, n_filtered_problems=None, n_twof_missing=0):
    write_manifest(
        summary_fpath,
        schema_version=SCHEMA_VERSION,
        status=status,
        detail=detail,
        metrics={'acc_clean': acc_clean, 'acc_2f': acc_2f, 'drag': drag,
                 'drag_threshold': float(config.drag_threshold)},
        # n_kept_problems is the PAIRED count -- the problems both accuracies
        # were measured on. n_filtered_problems is what the aggregate selected.
        # They differ exactly when the 2F round lost work, and reporting only
        # the latter overstates the evidence behind the drag.
        cohort={'n_kept_problems': n_kept_problems,
                'n_filtered_problems': (n_filtered_problems
                                        if n_filtered_problems is not None
                                        else n_kept_problems),
                'n_twof_missing': n_twof_missing},
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
