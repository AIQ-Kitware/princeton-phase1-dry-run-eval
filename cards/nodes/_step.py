"""
Shared helpers for the contextual-drag kwdagger nodes.

Each node shells out to the ``contextual_drag`` CLI exactly as the
monolithic card node did, so the science is unchanged -- only the
orchestration moves into the DAG.

Two frictions need bridging between that CLI and kwdagger:

*Globbed artifacts.*
    Steps write files whose names are only known after the fact
    (``evaluated_*_flattened.jsonl``). kwdagger needs a deterministic
    output path, so every node writes a small manifest naming the concrete
    artifacts it produced, and downstream nodes read that rather than
    re-globbing a directory they do not own.

*Steps that legitimately produce nothing.*
    The aggregate filter can remove every problem. That is a real result,
    not a failure, so the manifest records it and later nodes degrade
    rather than crash -- a DAG whose nodes must all succeed would
    otherwise turn a legible finding into a broken run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

__all__ = ['run_contextual_drag', 'write_manifest', 'read_manifest',
           'first_match']


def run_contextual_drag(args: list, check: bool = True) -> int:
    """
    Invoke the ``contextual_drag`` CLI in a subprocess.

    Args:
        args (list): arguments after the module name.
        check (bool): raise on a nonzero exit.

    Returns:
        int: the process return code.
    """
    command = [sys.executable, '-m', 'contextual_drag'] + [str(a) for a in args]
    print(f'[node] $ {" ".join(command)}', flush=True)
    completed = subprocess.run(command, check=check)
    return completed.returncode


def write_manifest(fpath, **payload) -> None:
    """
    Write a node's manifest.

    Args:
        fpath (str | Path): where to write it (the node's primary out path).
        **payload: recorded verbatim. Paths are stringified.
    """
    fpath = Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in payload.items()
    }
    with open(fpath, 'w') as file:
        json.dump(serializable, file, indent=2)
        file.write('\n')
    print(f'[node] wrote {fpath}', flush=True)


def read_manifest(fpath) -> dict:
    """Read an upstream node's manifest."""
    with open(fpath, 'r') as file:
        return json.load(file)


def first_match(dpath, pattern: str):
    """
    The single artifact matching ``pattern``, or None.

    Args:
        dpath (str | Path): directory to search.
        pattern (str): glob pattern.

    Returns:
        Path | None: the last match in sorted order, which is the most
            recent when names carry a timestamp.
    """
    matches = sorted(Path(dpath).glob(pattern))
    return matches[-1] if matches else None
