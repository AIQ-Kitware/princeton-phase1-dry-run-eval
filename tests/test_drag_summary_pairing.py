"""
The two accuracies in a drag must describe the same problems.

`cd_drag_summary`'s own docstring states the invariant: the headline number is
"a difference between two accuracies measured on the *same* set of problems".
It was not enforced. `acc_clean` was restricted to the problems the aggregate
filter kept, while `acc_2f` came from the evaluator's `overall_correctness` --
an average over whatever the 2F round managed to generate.

Those coincide only when the 2F round loses nothing. One overnight run on a
Turing host hit request timeouts, produced 2F output for 12 of 56 filtered
problems, and reported `drag=+0.3988` with `n_kept=56` -- a comparison between
a clean accuracy over 56 problems and a 2F accuracy over 12, labelled with the
larger number.
"""

import json

import pytest

from cards.nodes.cd_drag_summary import (
    _mean_over, _twof_accuracy_by_problem,
)


def _write_twof(dpath, per_problem):
    """Write an evaluated 2F dataset with a given correct/total per problem."""
    dpath.mkdir(parents=True, exist_ok=True)
    with open(dpath / 'evaluated_inference.jsonl', 'w') as file:
        for pid, (correct, total) in per_problem.items():
            gens = [{'correctness': i < correct} for i in range(total)]
            file.write(json.dumps({'id': pid, 'twof_generations': gens}) + '\n')
    return dpath


def test_twof_accuracy_is_per_problem_not_an_overall_average(tmp_path):
    """The 2F side must be addressable per problem, so it can be paired."""
    dpath = _write_twof(tmp_path / 'infer', {'a': (2, 4), 'b': (0, 4)})
    by_id = _twof_accuracy_by_problem(dpath)
    assert set(by_id) == {'a', 'b'}
    assert by_id['a'] == {'correct': 2, 'total': 4}


def test_pairing_excludes_problems_the_twof_round_never_produced(tmp_path):
    """
    A problem missing from the 2F round must not contribute to either side.

    This is the bug: `a` and `b` were answered in both rounds, `c` only in the
    clean round. Averaging the clean side over all three and the 2F side over
    the two that survived compares different cohorts.
    """
    clean = {'a': {'correct': 4, 'total': 4},
             'b': {'correct': 2, 'total': 4},
             'c': {'correct': 0, 'total': 4}}
    twof = _twof_accuracy_by_problem(
        _write_twof(tmp_path / 'infer', {'a': (1, 4), 'b': (1, 4)}))

    paired = sorted(set(clean) & set(twof))
    assert paired == ['a', 'b']

    acc_clean = _mean_over(clean, paired)
    acc_2f = _mean_over(twof, paired)
    assert acc_clean == pytest.approx(0.75)   # 6/8, not 6/12
    assert acc_2f == pytest.approx(0.25)

    # The unpaired form is what produced the bad number: it drags acc_clean
    # down with a problem the 2F round never answered.
    unpaired_clean = _mean_over(clean, sorted(clean))
    assert unpaired_clean == pytest.approx(0.5)
    assert unpaired_clean != acc_clean


def test_a_problem_with_no_generations_is_not_counted(tmp_path):
    """An empty 2F row is a lost problem, not a zero-accuracy problem."""
    dpath = tmp_path / 'infer'
    dpath.mkdir(parents=True)
    with open(dpath / 'evaluated_inference.jsonl', 'w') as file:
        file.write(json.dumps({'id': 'a', 'twof_generations': []}) + '\n')
        file.write(json.dumps(
            {'id': 'b', 'twof_generations': [{'correctness': True}]}) + '\n')
    by_id = _twof_accuracy_by_problem(dpath)
    assert set(by_id) == {'b'}, 'an empty row must not read as 0% correct'


def test_missing_dataset_dir_is_none(tmp_path):
    assert _twof_accuracy_by_problem(None) is None
    assert _twof_accuracy_by_problem(tmp_path / 'nope') is None
