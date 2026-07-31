"""
Sampling parameters that do not require vLLM to be installed.

``vllm.SamplingParams`` is the canonical type and is used verbatim when
vLLM is available, so a GPU run is byte-for-byte unchanged. When it is
not -- a REST endpoint run, or a machine without CUDA -- an equivalent
dataclass stands in, carrying exactly the fields this package sets.

Keeping the substitution here rather than at each call site means
``config.make_sampling_params`` builds the same object either way and no
caller has to care which one it got.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional

__all__ = ['SamplingParams', 'HAVE_VLLM', 'as_request_kwargs']

try:  # pragma: no cover - depends on the environment
    from vllm import SamplingParams  # type: ignore
    HAVE_VLLM = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_VLLM = False

    @dataclass
    class SamplingParams:  # type: ignore[no-redef]
        """
        Stand-in for ``vllm.SamplingParams``.

        Only the fields this package sets are declared. Anything vLLM
        accepts but we never pass is deliberately absent, so a typo in a
        model config raises here instead of being silently ignored.
        """

        n: int = 1
        seed: Optional[int] = None
        skip_special_tokens: bool = True
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        max_tokens: Optional[int] = None
        repetition_penalty: float = 1.0
        presence_penalty: float = 0.0
        frequency_penalty: float = 0.0

        def clone(self) -> 'SamplingParams':
            """Deep copy, matching ``vllm.SamplingParams.clone``.

            Callers that draw n samples as n separate requests clone the
            params and then mutate ``n`` and ``seed`` on the copy, so this
            must not alias the original.
            """
            import copy as _copy
            return _copy.deepcopy(self)


def as_request_kwargs(sampling_params: Any) -> dict:
    """
    Translate sampling params into an OpenAI completions request body.

    ``top_k``, ``repetition_penalty`` and ``skip_special_tokens`` are not
    OpenAI parameters. vLLM's own server accepts them as extensions, so
    they are passed through at the top level rather than dropped -- an
    endpoint that ignores them samples differently than the GPU path
    would, and silently dropping them would hide that.

    Args:
        sampling_params: a ``SamplingParams`` from either implementation.

    Returns:
        dict: keys suitable for a ``/v1/completions`` body.
    """
    def _get(name, default=None):
        return getattr(sampling_params, name, default)

    body: dict[str, Any] = {
        'n': int(_get('n', 1) or 1),
        'temperature': _get('temperature', 1.0),
        'top_p': _get('top_p', 1.0),
    }
    for name in ('max_tokens', 'seed', 'presence_penalty',
                 'frequency_penalty'):
        value = _get(name)
        if value is not None:
            body[name] = value
    # vLLM extensions -- meaningful to a vLLM-backed endpoint, ignorable
    # elsewhere, but never silently dropped.
    for name in ('top_k', 'repetition_penalty', 'skip_special_tokens'):
        value = _get(name)
        if value is not None:
            body[name] = value
    return body


def declared_fields() -> tuple[str, ...]:
    """Field names on whichever SamplingParams implementation is active."""
    if HAVE_VLLM:  # pragma: no cover - depends on the environment
        return tuple(
            sorted({f.name for f in fields(SamplingParams)})  # type: ignore
        ) if hasattr(SamplingParams, '__dataclass_fields__') else ()
    return tuple(f.name for f in fields(SamplingParams))  # type: ignore
