"""
The scheduler resolves endpoint aliases without importing contextual_drag.

`resolve_endpoints` runs at DAG-compile time, in the process that builds the
schedule -- not inside any node's container. That process has magnet and
kwdagger and nothing else. When the lookup imported `contextual_drag`, the card
compiled only on hosts where the package happened to be pip-installed next to
magnet, and failed everywhere else with a ModuleNotFoundError raised from the
middle of kwdagger's job submission.

Each test builds its own pipeline: ``model_config`` is a shared port wired from
init_inference to twof_inference and aggregate, so configuring one node fixes
the value for the others.
"""

import json

import pytest

from cards.pipelines import _model_params_fpath, drag_pipeline


def _inference_node(alias):
    node = drag_pipeline().node_dict['init_inference']
    node.configure({'model_config': alias})
    return node


def _blocks():
    return json.loads(_model_params_fpath().read_text())


def test_alias_resolves_to_the_served_model_name():
    """The endpoint alias is the config block's model_name, verbatim."""
    alias, block = next(
        (k, v) for k, v in _blocks().items() if v.get('model_name'))
    assert _inference_node(alias).resolve_endpoints() == [block['model_name']]


def test_the_scaleup_alias_resolves():
    """The alias contextual_drag_scaleup.yaml names must be addressable."""
    assert _inference_node('Gemma4_E2B').resolve_endpoints() == [
        'google/gemma-4-E2B-it']


def test_unknown_alias_yields_no_lease_rather_than_crashing():
    """The node reports a bad alias when it runs, naming the valid ones."""
    assert _inference_node('NotAModelThatExists').resolve_endpoints() == []


def test_missing_params_file_is_loud():
    """A broken checkout is never silently a zero-lease run."""
    with pytest.raises(RuntimeError):
        _model_params_fpath('/nonexistent/eval_models_params.json')


def test_lookup_does_not_import_contextual_drag(monkeypatch):
    """
    The regression itself: resolve with the package banned from import.

    Blocking the import outright is the only check that stays honest on a host
    where contextual_drag *is* installed -- which is exactly the kind of host
    the original bug hid on.
    """
    import builtins
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == 'contextual_drag' or name.startswith('contextual_drag.'):
            raise AssertionError(
                f'resolve_endpoints imported {name}; the scheduler must not '
                'depend on a node-container package')
        return real_import(name, *args, **kwargs)

    blocks = _blocks()
    alias = next(k for k, v in blocks.items() if v.get('model_name'))
    node = _inference_node(alias)

    monkeypatch.setattr(builtins, '__import__', guard)
    assert node.resolve_endpoints() == [blocks[alias]['model_name']]


def test_thinking_tristate_reaches_the_cli_as_a_flag():
    """
    `thinking` must round-trip from the card matrix to a contextual_drag flag.

    Passing neither --enable_thinking nor --disable_thinking is NOT neutral:
    contextual_drag's tri-state resolves "unset" to thinking ON. That is why
    the Qwen3_8B_NoThinking alias produced the same prompts as _Thinking.
    """
    from cards.nodes.cd_inference import CDInferenceCLI, _coerce_tristate

    cases = [(['--thinking=False'], False), (['--thinking=True'], True),
             ([], None)]
    for argv, expected in cases:
        config = CDInferenceCLI.cli(argv=argv, strict=True, verbose=False)
        assert _coerce_tristate(config.thinking) is expected


def test_the_card_asks_for_thinking_off_on_both_rounds():
    """A budget of 6144 tokens is not a fix if a preamble still eats it."""
    import pathlib
    import kwutil
    card = kwutil.Yaml.coerce(
        (pathlib.Path(__file__).parent.parent
         / 'cards/contextual_drag_scaleup.yaml').read_text())
    matrix = card['kwdagger']['matrix']
    for node in ('init_inference', 'twof_inference'):
        assert matrix[f'{node}.thinking'] is False
        assert matrix[f'{node}.max_tokens'] == 6144


def _claim_status(card_name, **symbols):
    """Run a card's claim through MAGNET's real evaluator."""
    import pathlib
    import kwutil
    from magnet.evaluation import Claim
    card = kwutil.Yaml.coerce(
        (pathlib.Path(__file__).parent.parent / 'cards' / card_name).read_text())
    return Claim({'python': card['claim']['python']}).evaluate(dict(symbols))[0]


@pytest.mark.parametrize('card_name', [
    'contextual_drag_kwdagger.yaml', 'contextual_drag_scaleup.yaml'])
def test_a_pipeline_that_produced_nothing_is_not_scored_as_falsified(card_name):
    """
    INCONCLUSIVE must not be reported as evidence against the claim.

    MAGNET maps AssertionError -> FALSIFIED and every other exception ->
    INCONCLUSIVE. The cards originally used `assert status != 'INCONCLUSIVE'`,
    so a run where every problem was filtered out -- which happened on namek,
    twice, with the same config -- was reported as refuting a TA1 team's claim
    rather than as having failed to test it.
    """
    assert _claim_status(
        card_name,
        status='INCONCLUSIVE', detail='every problem was filtered out',
        metrics={'drag': None, 'acc_clean': None, 'acc_2f': None},
        cohort={'n_kept_problems': 0}, drag_threshold=0.05,
    ) == 'INCONCLUSIVE'


@pytest.mark.parametrize('card_name', [
    'contextual_drag_kwdagger.yaml', 'contextual_drag_scaleup.yaml'])
def test_a_real_drag_below_threshold_is_still_falsified(card_name):
    """The fix must not soften an actual negative result."""
    assert _claim_status(
        card_name,
        status='OK', detail='',
        metrics={'drag': -0.0625, 'acc_clean': 0.4375, 'acc_2f': 0.5},
        cohort={'n_kept_problems': 2}, drag_threshold=0.05,
    ) == 'FALSIFIED'


@pytest.mark.parametrize('card_name', [
    'contextual_drag_kwdagger.yaml', 'contextual_drag_scaleup.yaml'])
def test_a_drag_above_threshold_verifies(card_name):
    assert _claim_status(
        card_name,
        status='OK', detail='',
        metrics={'drag': 0.20, 'acc_clean': 0.6, 'acc_2f': 0.4},
        cohort={'n_kept_problems': 12}, drag_threshold=0.05,
    ) == 'VERIFIED'
