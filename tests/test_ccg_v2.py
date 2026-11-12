"""Tests for the ccg_v2 card: the CCG v2 law, its scoring, and that the card generates nothing."""
import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest

from contextual_drag.analysis.ccg_v2.cli import run_from_files, score_payload
from contextual_drag.analysis.ccg_v2.model import ell, fit, line_baseline, predict, rung_accuracy

REPO_ROOT = Path(__file__).resolve().parents[1]
LADDERS = REPO_ROOT / 'data' / 'ccg_v2' / 'ladders'
CARD = REPO_ROOT / 'cards' / 'ccg_v2.yaml'

# Table 1 of "Contextual Drag as a Log-Count Vote" (Nov 11, 2026): c, w from T = 0, 1, 2 and the
# T = 3, T = 4 errors (pp).
NOTE = {
    'GPT_OSS_120B': (1.76, 1.27, -3.1, -0.2),
    'GPT_OSS_20B': (1.80, 0.93, -3.6, -0.4),
    'Gemma4_31B': (2.09, 1.06, +0.2, +0.7),
    'Gemma4_E4B': (1.75, 1.57, +0.7, +1.2),
    'Qwen3_32B': (1.24, 1.14, -0.2, -1.7),
    'Qwen3_8B': (1.67, 1.14, -3.2, -0.8),
}


def _payload(name):
    return json.loads((LADDERS / f'{name}.json').read_text())


def test_ell_is_antisymmetric_about_the_tie():
    for T in range(5):
        assert ell(T) == pytest.approx(-ell(4 - T))
    assert ell(2) == 0.0


def test_fit_recovers_parameters_of_an_exact_ladder():
    acc = {T: predict(0.4, 1.7, T) for T in range(5)}
    c, w = fit(acc, (0, 1, 2), 25)
    assert (c, w) == pytest.approx((0.4, 1.7))
    assert predict(c, w, 4) == pytest.approx(acc[4])


@pytest.mark.parametrize('name', sorted(NOTE))
def test_reproduces_the_note(name):
    c, w, e3, e4 = NOTE[name]
    out = score_payload(_payload(name), n_boot=0)
    assert out['c'] == pytest.approx(c, abs=0.006)
    assert out['w'] == pytest.approx(w, abs=0.006)
    assert out['err_pp']['3'] == pytest.approx(e3, abs=0.06)
    assert out['err_pp']['4'] == pytest.approx(e4, abs=0.06)
    assert out['pass'] is True


def test_line_baseline_uses_the_nearest_calibration_rungs():
    acc = {0: 0.2, 1: 0.5, 2: 0.7, 3: 0.0, 4: 0.0}
    assert line_baseline(acc, (0, 1, 2), 3) == pytest.approx(0.9)
    assert line_baseline(acc, (0, 1, 2), 4) == 1.0


def test_bootstrap_is_seeded():
    a = score_payload(_payload('Qwen3_8B'), n_boot=100, seed=3)
    b = score_payload(_payload('Qwen3_8B'), n_boot=100, seed=3)
    assert a['ci'] == b['ci']


def test_summary_over_bundled_ladders():
    out = run_from_files(sorted(LADDERS.glob('*.json')), n_boot=0)
    assert out['n_models'] == out['n_models_pass'] == 6
    assert out['max_abs_err_pp'] < 5.0
    json.dumps(out, allow_nan=False)


def test_bundled_ladders_match_provenance():
    text = (LADDERS / 'PROVENANCE.md').read_text()
    for f in LADDERS.glob('*.json'):
        assert hashlib.sha256(f.read_bytes()).hexdigest()[:16] in text, f.name


def _run_node(tmp_path, **kw):
    from cards.nodes.cd_ccg_v2 import CDCcgV2CLI
    summary = tmp_path / 'results.json'
    CDCcgV2CLI.main(argv=[], summary_fpath=str(summary), n_boot=20, **kw)
    return json.loads(summary.read_text())


def test_node_scores_the_bundled_ladders(tmp_path):
    result = _run_node(tmp_path, ladders_dpath=str(LADDERS))
    assert result['status'] == 'VERIFIED' and result['n_models_pass'] == 6


def test_node_scores_a_version1_store_layout(tmp_path):
    store = tmp_path / '_kwdagger' / 'ladder'
    for i, name in enumerate(('GPT_OSS_20B', 'Qwen3_32B')):
        d = store / f'ladder_id_{i}'
        d.mkdir(parents=True)
        shutil.copy(LADDERS / f'{name}.json', d / 'ladder.json')
    result = _run_node(tmp_path, ladders_dpath=str(store), ladders_glob='*/ladder.json',
                       n_targets_expected=2, min_models_pass=2)
    assert result['status'] == 'VERIFIED'
    assert sorted(result['per_model']) == ['GPT_OSS_20B', 'Qwen3_32B']


def test_node_is_inconclusive_without_ladders(tmp_path):
    empty = tmp_path / 'none'
    empty.mkdir()
    assert _run_node(tmp_path, ladders_dpath=str(empty))['status'] == 'INCONCLUSIVE'
    assert _run_node(tmp_path, ladders_dpath=str(LADDERS), n_targets_expected=5)['status'] == 'INCONCLUSIVE'


def test_card_generates_nothing():
    pytest.importorskip('kwdagger')
    import yaml
    from cards.pipelines import ccg_v2_pipeline
    nodes = [n.name for n in ccg_v2_pipeline().nodes]
    assert nodes == ['ccg_v2']
    matrix = yaml.safe_load(CARD.read_text())['kwdagger']['matrix']
    assert (REPO_ROOT / matrix['ccg_v2.ladders_dpath']).is_dir()
    assert not any('inference' in k for k in matrix)
