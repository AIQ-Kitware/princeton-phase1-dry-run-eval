"""
Score one inference round with the packaged math evaluator.

Used twice: after the clean round with ``--flatten_dataset`` (the
postprocess step needs the flattened form), and after the 2F round with
``--response_column twof_generations``.
"""

from __future__ import annotations

from pathlib import Path

import scriptconfig as scfg

from cards.nodes._step import (
    first_match, read_manifest, run_contextual_drag, write_manifest,
)


class CDEvalCLI(scfg.DataConfig):
    """Evaluate the generations produced by one inference node."""

    inference_manifest_fpath = scfg.Value(
        None, help='Manifest written by the inference node.',
        tags=['in_path'])

    flatten_dataset = scfg.Value(
        False, isflag=True,
        help='Emit the flattened jsonl the postprocess step consumes.',
        tags=['algo_param'])
    response_column = scfg.Value(
        None, help='Score this column instead of the default.',
        tags=['algo_param'])

    n_jobs = scfg.Value(1, type=int, help='Evaluator worker processes.',
                        tags=['perf_param'])

    manifest_fpath = scfg.Value(
        'eval.json', help='Manifest naming the evaluation artifacts.',
        tags=['out_path', 'primary'])

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        manifest_fpath = Path(config.manifest_fpath).resolve()
        upstream = read_manifest(config.inference_manifest_fpath)

        if upstream.get('skipped'):
            write_manifest(manifest_fpath, skipped=True,
                           reason=upstream.get('reason'),
                           flattened_fpath=None, error_analysis_fpath=None,
                           dataset_dir=None)
            return

        dataset_dir = Path(upstream['output_dir'])
        args = ['eval', 'math', '--dataset_dir', dataset_dir,
                '--single_partition', '--n_jobs', config.n_jobs]
        if config.flatten_dataset:
            args.append('--flatten_dataset')
        if config.response_column:
            args += ['--response_column', config.response_column]
        run_contextual_drag(args)

        write_manifest(
            manifest_fpath,
            skipped=False,
            dataset_dir=dataset_dir,
            flattened_fpath=first_match(dataset_dir, 'evaluated_*_flattened.jsonl'),
            error_analysis_fpath=first_match(
                dataset_dir, 'evaluated_*_error_analysis.json'),
        )


__cli__ = CDEvalCLI

if __name__ == '__main__':
    CDEvalCLI.main()
