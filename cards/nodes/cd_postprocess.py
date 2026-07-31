"""
Fold the flattened clean-round generations into a dataset.

Thin wrapper over ``contextual_drag data initial-sampling-postprocess``.
"""

from __future__ import annotations

from pathlib import Path

import scriptconfig as scfg

from cards.nodes._step import (
    read_manifest, run_contextual_drag, write_manifest,
)


class CDPostprocessCLI(scfg.DataConfig):
    """Build the processed dataset the aggregate step filters."""

    eval_manifest_fpath = scfg.Value(
        None, help='Manifest written by the clean-round eval node.',
        tags=['in_path'])

    manifest_fpath = scfg.Value(
        'postprocess.json', help='Manifest naming the processed dataset.',
        tags=['out_path', 'primary'])

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        manifest_fpath = Path(config.manifest_fpath).resolve()
        upstream = read_manifest(config.eval_manifest_fpath)

        if upstream.get('skipped') or not upstream.get('flattened_fpath'):
            write_manifest(manifest_fpath, skipped=True,
                           reason='no flattened generations upstream',
                           processed_ds=None)
            return

        flattened = Path(upstream['flattened_fpath'])
        work_dir = manifest_fpath.parent
        # The CLI globs relative to input_dir; point it straight at the file
        # the eval node reported rather than re-deriving a directory layout.
        run_contextual_drag([
            'data', 'initial-sampling-postprocess',
            '--input_dir', flattened.parent.parent,
            '--input_file_template', f'{flattened.parent.name}/{flattened.name}',
        ])

        processed = flattened.parent.parent / 'processed_flattened_init_responses.ds'
        if not processed.exists():
            raise FileNotFoundError(
                f'postprocess did not create {processed}')
        write_manifest(manifest_fpath, skipped=False, processed_ds=processed,
                       work_dir=work_dir)


__cli__ = CDPostprocessCLI

if __name__ == '__main__':
    CDPostprocessCLI.main()
