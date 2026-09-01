"""
Result loading for this team's kwdagger nodes.

kwdagger has no generic ``load_result`` for a Python ``ProcessNode`` -- the base
class documents it as the extension point and ``aggregate_loader`` requires a
flat dot-dictionary, so a node without one surfaces as
``AttributeError: 'dict' object has no attribute 'query_keys'`` when MAGNET asks
kwdagger for the available result rows. (``YamlProcessNode`` ships one; classes
declared in Python do not inherit it.)

This is that loader, shared by every node here.
"""
import json

import ubelt as ub


def load_node_result(node, node_dpath):
    """
    Load one node's primary artifact into the aggregate namespaces.

    The artifact is read from the node's ``primary_out_key``. Everything in it
    except ``info`` becomes ``metrics.<node>.*``; ``info[-1]`` is the
    ``kwutil.ProcessContext`` block and supplies ``resolved_params.<node>.*``.
    An artifact that nests its payload under ``result`` is unwrapped, so both
    the enveloped and the flat shapes load the same way.

    Args:
        node (kwdagger.ProcessNode): the node being loaded.
        node_dpath (str | PathLike): its output directory.

    Returns:
        DotDict: flat, prefixed with the node name.
    """
    from kwdagger.aggregate_loader import new_process_context_parser
    from kwdagger.utils import util_dotdict

    node_dpath = ub.Path(node_dpath)
    payload = json.loads(
        (node_dpath / node.out_paths[node.primary_out_key]).read_text())

    nested = {}
    info = payload.get("info")
    if info:
        nested.update(new_process_context_parser(info[-1]))

    body = payload.get("result", payload)
    # The artifact's own shape is preserved. Lifting a nested `metrics` key up
    # would be convenient for cards that read `metrics.<node>.mae`, but it
    # destroys the sub-dict that cards doing `globals().update(...)` then index
    # as `metrics['mae']`. A node that wants flat names writes flat names.
    nested["metrics"] = {
        key: value for key, value in body.items()
        if key != "info" and not key.startswith("_")
    }

    flat = util_dotdict.DotDict.from_nested(nested)
    return flat.insert_prefix(node.name, index=1)
