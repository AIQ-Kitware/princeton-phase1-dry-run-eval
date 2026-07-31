"""
kwdagger pipelines for the contextual-drag cards.

The card previously ran the whole six-step chain inside one node, which
meant any change re-ran everything: adjust the aggregate filter and the
expensive clean-round inference was recomputed even though its inputs were
untouched. As a DAG each stage caches on its own identity, so a filter
sweep reuses the generations.

    init_inference          <- clean prompt, the expensive step
          |
    eval_init               (--flatten_dataset)
          |
    postprocess
          |
    aggregate               <- may legitimately keep nothing
          |
    twof_inference          <- 2F prompt, the second expensive step
          |
    eval_twof               (--response_column twof_generations)
          |
    drag_summary            <- terminal artifact

``drag_summary`` takes two inputs, not one: the difference it reports is
between accuracies measured on the same surviving problems, so it needs
the aggregate's dataset as well as the 2F evaluation.
"""

from __future__ import annotations

import kwdagger

from cards.nodes.cd_aggregate import CDAggregateCLI
from cards.nodes.cd_drag_summary import CDDragSummaryCLI
from cards.nodes.cd_eval import CDEvalCLI
from cards.nodes.cd_inference import CDInferenceCLI
from cards.nodes.cd_postprocess import CDPostprocessCLI

__all__ = ['drag_pipeline']


class _InitInference(kwdagger.ProcessNode):
    """Clean-prompt generation."""
    name = 'init_inference'
    executable = 'python -m cards.nodes.cd_inference'
    params = CDInferenceCLI


class _TwofInference(kwdagger.ProcessNode):
    """2F-augmented generation."""
    name = 'twof_inference'
    executable = 'python -m cards.nodes.cd_inference'
    params = CDInferenceCLI


class _EvalInit(kwdagger.ProcessNode):
    """Score the clean round, emitting the flattened form."""
    name = 'eval_init'
    executable = 'python -m cards.nodes.cd_eval'
    params = CDEvalCLI


class _EvalTwof(kwdagger.ProcessNode):
    """Score the 2F round."""
    name = 'eval_twof'
    executable = 'python -m cards.nodes.cd_eval'
    params = CDEvalCLI


class _Postprocess(kwdagger.ProcessNode):
    """Fold flattened generations into a dataset."""
    name = 'postprocess'
    executable = 'python -m cards.nodes.cd_postprocess'
    params = CDPostprocessCLI


class _Aggregate(kwdagger.ProcessNode):
    """Filter problems and build the 2F dataset."""
    name = 'aggregate'
    executable = 'python -m cards.nodes.cd_aggregate'
    params = CDAggregateCLI


class _DragSummary(kwdagger.ProcessNode):
    """Compute the drag and emit the terminal artifact."""
    name = 'drag_summary'
    executable = 'python -m cards.nodes.cd_drag_summary'
    params = CDDragSummaryCLI


def drag_pipeline():
    """
    Build the contextual-drag DAG.

    Returns:
        kwdagger.Pipeline

    Example:
        >>> from cards.pipelines import drag_pipeline
        >>> sorted(drag_pipeline().nodes)
        ['aggregate', 'drag_summary', 'eval_init', 'eval_twof', 'init_inference', 'postprocess', 'twof_inference']
    """
    nodes = {
        'init_inference': _InitInference(),
        'eval_init': _EvalInit(),
        'postprocess': _Postprocess(),
        'aggregate': _Aggregate(),
        'twof_inference': _TwofInference(),
        'eval_twof': _EvalTwof(),
        'drag_summary': _DragSummary(),
    }

    nodes['init_inference'].outputs['manifest_fpath'].connect(
        nodes['eval_init'].inputs['inference_manifest_fpath'])
    nodes['eval_init'].outputs['manifest_fpath'].connect(
        nodes['postprocess'].inputs['eval_manifest_fpath'])
    nodes['postprocess'].outputs['manifest_fpath'].connect(
        nodes['aggregate'].inputs['postprocess_manifest_fpath'])

    # The aggregate's manifest names the 2F dataset, so it is the second
    # round's input as well as the summary's.
    nodes['aggregate'].outputs['manifest_fpath'].connect(
        nodes['twof_inference'].inputs['data_fpath'])
    nodes['twof_inference'].outputs['manifest_fpath'].connect(
        nodes['eval_twof'].inputs['inference_manifest_fpath'])

    nodes['aggregate'].outputs['manifest_fpath'].connect(
        nodes['drag_summary'].inputs['aggregate_manifest_fpath'])
    nodes['eval_twof'].outputs['manifest_fpath'].connect(
        nodes['drag_summary'].inputs['twof_eval_manifest_fpath'])

    # The two rounds share a model. Wire it rather than restating it, so a
    # model sweep is declared once.
    nodes['init_inference'].param_ports['model_config'].connect(
        nodes['twof_inference'].param_ports['model_config'])
    nodes['init_inference'].param_ports['model_config'].connect(
        nodes['aggregate'].param_ports['model_config'])

    dag = kwdagger.Pipeline(nodes)
    dag.build_nx_graphs()
    return dag
