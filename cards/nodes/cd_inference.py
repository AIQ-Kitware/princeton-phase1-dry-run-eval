"""
One inference round: render prompts and generate responses.

Used twice in the drag pipeline -- once on the clean prompt and once on
the 2F-augmented prompt -- as two nodes sharing this executable.

``gpu_memory_utilization`` and ``tensor_parallel_size`` are perf params,
not algo params. They change how the work is placed on hardware, never
what is produced, so they must not enter the node's identity: raising the
memory fraction should not invalidate generations. That was wrong in the
monolithic card, where ``gpu_memory_utilization`` was declared an algo
param and so was part of the card's hash. They are also ignored entirely
when generation goes through a REST endpoint.
"""

from __future__ import annotations

from pathlib import Path

import scriptconfig as scfg

from cards.nodes._step import (
    first_match, read_manifest, run_contextual_drag, write_manifest,
)


class CDInferenceCLI(scfg.DataConfig):
    """Run one inference cell."""

    model_config = scfg.Value(
        'Qwen3_8B_NoThinking',
        help='Alias from eval_models_params.json.',
        tags=['algo_param'])

    data_fpath = scfg.Value(
        None,
        help=('Dataset to run on. Either a .ds path (the clean round) or an '
              'upstream manifest naming one (the 2F round).'),
        tags=['in_path'])

    template_path = scfg.Value(
        'prompt_templates/init_response_prompt_templates.json',
        help='JSON file of prompt templates.', tags=['algo_param'])
    template_key = scfg.Value(
        'qa_mc_prompt', help='Template key within that file.',
        tags=['algo_param'])
    task_name = scfg.Value(
        'init_response', help='Names the generated columns.',
        tags=['algo_param'])

    max_questions = scfg.Value(
        8, type=int, help='Cap on dataset rows (0 = whole dataset).',
        tags=['algo_param'])
    n = scfg.Value(8, type=int, help='Samples per question.',
                   tags=['algo_param'])
    max_tokens = scfg.Value(2048, type=int, help='Per-response budget.',
                            tags=['algo_param'])

    # Placement, not behaviour -- see the module docstring.
    gpu_memory_utilization = scfg.Value(
        0.85, type=float, help='vLLM GPU memory fraction. Ignored on a REST '
                               'endpoint.', tags=['perf_param'])
    tensor_parallel_size = scfg.Value(
        1, type=int, help='vLLM tensor parallelism. Ignored on a REST '
                          'endpoint.', tags=['perf_param'])

    manifest_fpath = scfg.Value(
        'inference.json', help='Manifest naming the completions produced.',
        tags=['out_path', 'primary'])

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)

        manifest_fpath = Path(config.manifest_fpath).resolve()
        output_dir = manifest_fpath.parent / 'inference'
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_fpath, n_rows = _resolve_dataset(config.data_fpath)
        if dataset_fpath is None:
            # Upstream produced nothing usable. Record it and stop; the
            # summary node turns this into a legible card outcome.
            write_manifest(manifest_fpath, dataset_fpath=None, n_rows=0,
                           completions_fpath=None, output_dir=output_dir,
                           task_name=config.task_name, skipped=True,
                           reason='upstream produced no usable dataset')
            return

        max_questions = int(config.max_questions)
        if n_rows is not None and n_rows > 0:
            max_questions = min(max_questions, n_rows) if max_questions else n_rows

        run_contextual_drag([
            'inference', 'run',
            '--model_config', config.model_config,
            '--data_path', dataset_fpath,
            '--prompt_template_path', config.template_path,
            '--prompt_template_key', config.template_key,
            '--output_dir', output_dir,
            '--task_name', config.task_name,
            '--max_questions', max_questions,
            '--n', config.n,
            '--batch_size', min(8, max(1, max_questions)),
            '--tensor_parallel_size', config.tensor_parallel_size,
            '--gpu_memory_utilization', config.gpu_memory_utilization,
            '--max_tokens', config.max_tokens,
        ])

        completions = first_match(output_dir, 'completions.jsonl')
        write_manifest(manifest_fpath, dataset_fpath=dataset_fpath,
                       n_rows=max_questions, completions_fpath=completions,
                       output_dir=output_dir, task_name=config.task_name,
                       skipped=False)


def _resolve_dataset(data_fpath):
    """
    Resolve the input, which is either a dataset or an upstream manifest.

    Returns:
        tuple: ``(dataset_fpath | None, n_rows | None)``.
    """
    path = Path(data_fpath)
    if path.is_dir():
        return path, None
    manifest = read_manifest(path)
    dataset = manifest.get('dataset_fpath')
    return (Path(dataset) if dataset else None), manifest.get('n_kept')


__cli__ = CDInferenceCLI

if __name__ == '__main__':
    CDInferenceCLI.main()
