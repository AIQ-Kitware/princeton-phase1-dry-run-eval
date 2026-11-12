"""
The only node of CCG v2: score existing ladders with the CCG v2 law.

Reads every ladder file matching ``ladders_glob`` under ``ladders_dpath`` -- by default the six
bundled competition-mathematics ladders in ``data/ccg_v2/ladders``, or the ``ladder`` outputs of a
version-1 run (``<OUTPUT_PATH>/_kwdagger/ladder`` with glob ``*/ladder.json``) --
runs ``contextual_drag analysis ccg_v2 score``, and emits the card's whole result. Each model
is fitted on its own T = 0, 1, 2 and its T = 3 and T = 4 accuracies are predicted. Nothing is
generated, so the card needs no GPU and no model endpoint.

The verdict is the claim's to make. This node reports the errors, the intervals and the same pass
rule, and reports INCONCLUSIVE when the ladders cannot support a verdict: no ladder found, a
ladder not marked OK, or a different number of models than the card declares.
"""
from __future__ import annotations

import json
from pathlib import Path

import scriptconfig as scfg

from cards.nodes._step import run_contextual_drag, write_manifest
import magnet.theory as theory

#: Bump when the terminal-result shape changes incompatibly.
SCHEMA_VERSION = 1


@theory.satisfies('Hygiene.Inference.threshold_exceeds_sampling_error::hprespecified',
                  note='tolerance_pp, the calibration and predicted rungs and min_models_pass are '
                       'card defaults fixed before the run')
class CDCcgV2CLI(scfg.DataConfig):
    """Score existing ladders with the CCG v2 law (calibrate T = 0, 1, 2; predict T = 3, 4)."""

    ladders_dpath = scfg.Value(None, help='Directory holding the ladder files.', tags=['in_path'])
    ladders_glob = scfg.Value('*.json', type=str, help='Glob under ladders_dpath.', tags=['algo_param'])
    n_targets_expected = scfg.Value(6, type=int, help='Models the card scores. Fewer is INCONCLUSIVE.',
                                    tags=['algo_param'])
    calibration_rungs = scfg.Value('0,1,2', type=str, tags=['algo_param'])
    predict_rungs = scfg.Value('3,4', type=str, tags=['algo_param'])
    tolerance_pp = scfg.Value(5.0, type=float, tags=['algo_param'])
    min_models_pass = scfg.Value(6, type=int, tags=['algo_param'])
    n_boot = scfg.Value(2000, type=int, tags=['algo_param'])
    boot_seed = scfg.Value(20261111, type=int, tags=['algo_param'])
    confidence = scfg.Value(0.95, type=float, tags=['algo_param'])

    summary_fpath = scfg.Value('results.json', help="The card's terminal artifact.", tags=['out_path', 'primary'])

    @theory.tests('AIQ.Teams.Princeton.CCGv2.lower_half_predicts_upper')
    @theory.assumes('AIQ.Teams.Princeton.CCGv2.lower_half_predicts_upper::hlaw',
                    note='one (c, w) per model on a task, common across problems')
    @theory.satisfies('AIQ.Teams.Princeton.CCGv2.lower_half_predicts_upper::hself',
                      note='every ladder was measured with drafts the model wrote itself')
    @theory.satisfies('AIQ.Teams.Princeton.CCGv2.lower_half_predicts_upper::hheldout',
                      note='only T = 0, 1, 2 enter the fit; T = 3, 4 never do, and no other model contributes')
    @theory.tests('Hygiene.Inference.threshold_exceeds_sampling_error')
    @theory.assumes('Hygiene.Inference.threshold_exceeds_sampling_error::hn_sufficient',
                    note='18 to 30 eligible problems per model; bootstrap intervals are reported and are '
                         'expected to be wider than 5 points at T = 3')
    @theory.approximates('Hygiene.Measurement.measured_score_tracks_construct')
    @theory.approximates('Hygiene.Measurement.measured_score_tracks_construct::hscorer',
                         note='the last boxed answer, compared to the key with math_verify')
    @theory.assumes('Hygiene.Measurement.measured_score_tracks_construct::hcontam',
                    note='AIME and HMMT 2024-25 are public; no contamination check')
    @theory.assumes('Hygiene.Measurement.measured_score_tracks_construct::hstable',
                    note='the bundled ladders were generated with fixed seeds and orders; the card does '
                         'not regenerate them')
    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        summary_fpath = Path(config.summary_fpath).resolve()
        dpath = Path(config.ladders_dpath)
        files = sorted(dpath.glob(config.ladders_glob)) if dpath.is_dir() else []
        ladders, usable = {}, []
        for f in files:
            payload = json.loads(f.read_text())
            ok = payload.get('status', 'OK') == 'OK' and bool(payload.get('problems'))
            ladders[str(payload.get('model', f.stem))] = {
                'status': payload.get('status', 'OK'), 'detail': payload.get('detail', ''),
                'n_problems': len(payload.get('problems', [])), 'ladder_fpath': str(f)}
            if ok:
                usable.append(f)
        problems = [f'{m}: {s["detail"] or s["status"]}' for m, s in sorted(ladders.items())
                    if s['status'] != 'OK' or not s['n_problems']]
        if not files or problems or len(usable) != int(config.n_targets_expected):
            detail = '; '.join(problems) or (f'{len(usable)} usable ladders under {dpath}/{config.ladders_glob}; '
                                             f'the card declares {config.n_targets_expected}')
            _emit(summary_fpath, status='INCONCLUSIVE', detail=detail, ladders=ladders)
            return
        analysis_fpath = summary_fpath.parent / 'ccg_v2_scores.json'
        run_contextual_drag(['analysis', 'ccg_v2', 'score', '--ladders', *usable,
                             '--calibration_rungs', config.calibration_rungs, '--predict_rungs', config.predict_rungs,
                             '--tolerance_pp', config.tolerance_pp, '--n_boot', config.n_boot,
                             '--seed', config.boot_seed, '--confidence', config.confidence, '--out', analysis_fpath])
        analysis = json.loads(analysis_fpath.read_text())
        status = 'VERIFIED' if analysis['n_models_pass'] >= int(config.min_models_pass) else 'FALSIFIED'
        _emit(summary_fpath, status=status, detail='', ladders=ladders, analysis=analysis,
              analysis_fpath=analysis_fpath, min_models_pass=int(config.min_models_pass))


def _emit(summary_fpath, *, status, detail, ladders, analysis=None, analysis_fpath=None, min_models_pass=None):
    analysis = analysis or {}
    keys = ('n_models', 'n_models_pass', 'n_models_supported', 'n_models_line_pass', 'mae_pp',
            'max_abs_err_pp', 'line_mae_pp', 'tolerance_pp')
    write_manifest(summary_fpath, schema_version=SCHEMA_VERSION, status=status, detail=detail,
                   **{k: analysis.get(k) for k in keys}, min_models_pass=min_models_pass,
                   per_model=analysis.get('per_model', {}), ladders=ladders, config=analysis.get('config'),
                   analysis_fpath=analysis_fpath)
    print(f'[ccg_v2] status={status} n_models_pass={analysis.get("n_models_pass")} '
          f'mae_pp={analysis.get("mae_pp")} {detail}', flush=True)


__cli__ = CDCcgV2CLI

if __name__ == '__main__':
    CDCcgV2CLI.main()
