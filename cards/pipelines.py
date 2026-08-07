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

import json
from pathlib import Path

import kwdagger
from magnet.containers import ContainerProcessNode
from magnet.leasing import LeasedProcessNode

from cards.nodes.cd_aggregate import CDAggregateCLI
from cards.nodes.cd_drag_summary import CDDragSummaryCLI
from cards.nodes.cd_eval import CDEvalCLI
from cards.nodes.cd_inference import CDInferenceCLI
from cards.nodes.cd_postprocess import CDPostprocessCLI

__all__ = ['drag_pipeline']

# eval_models_params.json as it sits in the checkout. `resolve_endpoints` runs
# in the *scheduler*, which is the one process in this pipeline that has only
# magnet + kwdagger -- every node's own dependency is satisfied inside its
# container. Importing ``contextual_drag`` to read this mapping therefore made
# DAG compilation depend on a package the scheduler has no reason to have, and
# the card crashed with ModuleNotFoundError on any host where it was not also
# pip-installed alongside magnet. It is a JSON file; read it as one.
_MODEL_PARAMS_RELPATH = Path(
    'src/contextual_drag/resources/inference/eval_models_params.json')


def _model_params_fpath(override=None) -> Path:
    """
    Locate eval_models_params.json without importing ``contextual_drag``.

    Args:
        override (str | None): explicit path, when a node names one.

    Returns:
        Path

    Raises:
        RuntimeError: if the packaged resource cannot be found. That is a
            broken checkout, not a card mistake, so it must be loud.
    """
    if override:
        candidate = Path(override).expanduser()
        if candidate.exists():
            return candidate
        raise RuntimeError(
            f'model params file {candidate} does not exist')
    # cards/ sits directly under the repo root, next to src/.
    candidate = Path(__file__).resolve().parent.parent / _MODEL_PARAMS_RELPATH
    if candidate.exists():
        return candidate
    # Installed layout (no checkout): fall back to the packaged copy.
    try:
        from importlib import resources
        packaged = Path(str(resources.files(
            'contextual_drag.resources.inference'
        ).joinpath('eval_models_params.json')))
        if packaged.exists():
            return packaged
    except Exception:
        pass
    raise RuntimeError(
        f'cannot locate eval_models_params.json; looked for {candidate} and '
        'for the packaged contextual_drag.resources.inference copy. The card '
        'cannot resolve which endpoint its generation rounds need.')


class _Inference(LeasedProcessNode):
    """
    A generation round, holding its model only while it generates.

    The card names a *model config* (``Qwen3_8B_NoThinking``), not a served
    model, so the endpoint alias has to be looked up rather than read
    straight off a parameter. The alias is the config's ``model_name`` --
    the same string the REST engine sends as ``model`` -- so an endpoint
    that serves this card is one the card can already address.
    """

    executable = 'python -m cards.nodes.cd_inference'
    params = CDInferenceCLI

    def resolve_endpoints(self):
        config = self.final_config or {}
        alias = config.get('model_config')
        if not alias:
            return []
        fpath = _model_params_fpath(config.get('model_params_fpath'))
        blocks = json.loads(fpath.read_text())
        block = blocks.get(alias)
        if block is None:
            # An unknown alias is the node's problem to report when it runs,
            # with its own message naming the available aliases. Raising here
            # would turn it into an opaque DAG-compile failure. A *missing
            # file* is different and does raise, above -- that one is never
            # the card's fault.
            return []
        served = block.get('model_name')
        return [served] if served else []


class _InitInference(_Inference):
    """Clean-prompt generation."""
    name = 'init_inference'


class _TwofInference(_Inference):
    """2F-augmented generation."""
    name = 'twof_inference'


class _EvalInit(ContainerProcessNode):
    """Score the clean round, emitting the flattened form."""
    name = 'eval_init'
    executable = 'python -m cards.nodes.cd_eval'
    params = CDEvalCLI


class _EvalTwof(ContainerProcessNode):
    """Score the 2F round."""
    name = 'eval_twof'
    executable = 'python -m cards.nodes.cd_eval'
    params = CDEvalCLI


class _Postprocess(ContainerProcessNode):
    """Fold flattened generations into a dataset."""
    name = 'postprocess'
    executable = 'python -m cards.nodes.cd_postprocess'
    params = CDPostprocessCLI


class _Aggregate(ContainerProcessNode):
    """Filter problems and build the 2F dataset."""
    name = 'aggregate'
    executable = 'python -m cards.nodes.cd_aggregate'
    params = CDAggregateCLI


class _DragSummary(ContainerProcessNode):
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

    dag = kwdagger.Pipeline(list(nodes.values()))
    dag.build_nx_graphs()
    return dag
