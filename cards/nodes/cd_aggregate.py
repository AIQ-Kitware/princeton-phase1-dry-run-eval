"""
Build the 2F dataset by injecting trajectories into each problem.

This is the step that can legitimately produce nothing: the keep-filter
drops problems without enough correct and incorrect samples, and on a
small or easy dataset that can remove all of them. That is a real finding
about the cohort, not a failure, so it is recorded in the manifest and the
pipeline continues to a legible verdict rather than erroring out.
"""

from __future__ import annotations

from pathlib import Path

import scriptconfig as scfg

from cards.nodes._step import (
    read_manifest, run_contextual_drag, write_manifest,
)


class CDAggregateCLI(scfg.DataConfig):
    """Filter problems and build the 2F-augmented dataset."""

    postprocess_manifest_fpath = scfg.Value(
        None, help='Manifest written by the postprocess node.',
        tags=['in_path'])

    model_config = scfg.Value(
        'Qwen3_8B_NoThinking',
        help='Recorded as the init-response model on the aggregate.',
        tags=['algo_param'])

    num_true = scfg.Value(
        0, type=int, help='Correct trajectories injected per 2F prompt.',
        tags=['algo_param'])
    num_false = scfg.Value(
        2, type=int, help='Failed trajectories injected per 2F prompt.',
        tags=['algo_param'])
    min_num_true_sampling = scfg.Value(
        2, type=int,
        help='Keep a problem only with >= this many correct responses.',
        tags=['algo_param'])
    min_num_false_sampling = scfg.Value(
        2, type=int,
        help='Keep a problem only with >= this many incorrect responses.',
        tags=['algo_param'])

    manifest_fpath = scfg.Value(
        'aggregate.json', help='Manifest naming the 2F dataset, if any.',
        tags=['out_path', 'primary'])

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        manifest_fpath = Path(config.manifest_fpath).resolve()
        upstream = read_manifest(config.postprocess_manifest_fpath)

        if upstream.get('skipped') or not upstream.get('processed_ds'):
            write_manifest(manifest_fpath, skipped=True, n_kept=0,
                           dataset_fpath=None, processed_ds=None,
                           reason='no processed dataset upstream')
            return

        processed_ds = Path(upstream['processed_ds'])
        output_dir = manifest_fpath.parent / 'aggregate'
        output_dir.mkdir(parents=True, exist_ok=True)

        # check=False: a nonzero exit here means "nothing survived the
        # filter", which the manifest records rather than raising.
        returncode = run_contextual_drag([
            'data', 'aggregate',
            '--input_dir', processed_ds,
            '--num_true', config.num_true,
            '--num_false', config.num_false,
            '--min_num_true_sampling', config.min_num_true_sampling,
            '--min_num_false_sampling', config.min_num_false_sampling,
            '--output_dir', output_dir,
            '--init_response_models', config.model_config,
        ], check=False)

        twof_ds = (output_dir /
                   f'minimal_aggregated_data_T{config.num_true}'
                   f'_F{config.num_false}.ds')
        usable = (twof_ds / 'dataset_info.json').exists()

        if returncode != 0 or not usable:
            write_manifest(
                manifest_fpath, skipped=True, n_kept=0, dataset_fpath=None,
                processed_ds=processed_ds,
                reason=(f'aggregate produced no usable dataset '
                        f'(exit {returncode}); every problem was filtered out'))
            return

        write_manifest(manifest_fpath, skipped=False,
                       dataset_fpath=twof_ds, processed_ds=processed_ds,
                       n_kept=_len_dataset(twof_ds))


def _len_dataset(dpath) -> int:
    from datasets import load_from_disk
    return len(load_from_disk(str(dpath)))


__cli__ = CDAggregateCLI

if __name__ == '__main__':
    CDAggregateCLI.main()
