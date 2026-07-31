"""
An OpenAI-compatible REST engine, drop-in for ``AsyncLLMEngine``.

The package has always constructed its own in-process vLLM engine, which
means a card run needs an idle GPU to itself, cannot share inference with
anything else, and cannot be exercised at all without CUDA. This provides
the same interface over HTTP so the model can live anywhere -- a shared
vLLM server, a routed deployment, or a deterministic mock.

The contract the rest of this package relies on is small::

    async for output in engine.generate(prompt, sampling_params,
                                        request_id=...):
        ...
    await engine.abort(request_id)

where the final output has ``.finished`` true, ``.outputs`` (each with
``.text`` and ``.finish_reason``) and ``.prompt_token_ids``. That is what
is implemented here, and nothing else -- this is a transport, not a
reimplementation of vLLM.

``/v1/completions`` is used rather than ``/v1/chat/completions`` because
prompts reach the engine already rendered: whatever templating a task
needs has happened upstream, and the engine is handed a finished string.
Sending it as a chat message would let the server apply a chat template a
second time.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from contextual_drag.inference.sampling import as_request_kwargs

__all__ = ['EndpointEngine', 'EndpointConfig', 'endpoint_from_env']

#: Base URL of an OpenAI-compatible endpoint, e.g. http://127.0.0.1:8137/v1
ENDPOINT_ENVVAR = 'CONTEXTUAL_DRAG_ENDPOINT'
#: Model name to request, when it should differ from the config's model_name.
ENDPOINT_MODEL_ENVVAR = 'CONTEXTUAL_DRAG_ENDPOINT_MODEL'
#: Optional bearer token.
ENDPOINT_KEY_ENVVAR = 'CONTEXTUAL_DRAG_ENDPOINT_API_KEY'


@dataclass
class _Completion:
    """
    One returned sample. Mirrors a vLLM ``CompletionOutput``.

    ``token_ids`` is empty rather than fabricated. A completions endpoint
    reports token usage for the response as a whole, not per choice, and
    inventing ids to make a throughput counter look populated would be
    lying about what was measured. The consequence is that token/s reads
    as zero on an endpoint-backed run.
    """

    index: int
    text: str
    finish_reason: Optional[str]
    token_ids: list = field(default_factory=list)


@dataclass
class _RequestOutput:
    """
    Mirrors a vLLM ``RequestOutput`` closely enough for ``build_record``.

    ``prompt_token_ids`` is empty rather than fabricated: an OpenAI
    endpoint does not return them, and inventing plausible ids would make
    downstream token accounting quietly wrong instead of visibly absent.
    """

    outputs: list
    prompt_token_ids: list = field(default_factory=list)
    finished: bool = True


@dataclass
class EndpointConfig:
    """How to reach an OpenAI-compatible endpoint."""

    base_url: str
    model: str
    api_key: Optional[str] = None
    timeout: float = 600.0
    max_retries: int = 3

    def completions_url(self) -> str:
        return self.base_url.rstrip('/') + '/completions'


def endpoint_from_env(model_config: dict) -> Optional[EndpointConfig]:
    """
    Build an :class:`EndpointConfig` from the environment, if configured.

    Args:
        model_config (dict): the resolved model config block, used for its
            ``model_name`` unless overridden.

    Returns:
        EndpointConfig | None: None when no endpoint is configured, in
            which case the caller should build a local vLLM engine.
    """
    base_url = os.environ.get(ENDPOINT_ENVVAR, '').strip()
    if not base_url:
        return None
    model = (
        os.environ.get(ENDPOINT_MODEL_ENVVAR, '').strip()
        or model_config.get('model_name')
    )
    if not model:
        raise RuntimeError(
            f'{ENDPOINT_ENVVAR} is set but no model name is available; set '
            f'{ENDPOINT_MODEL_ENVVAR} or give the model config a '
            f'"model_name".'
        )
    return EndpointConfig(
        base_url=base_url,
        model=model,
        api_key=os.environ.get(ENDPOINT_KEY_ENVVAR) or None,
    )


class EndpointEngine:
    """
    Serve ``generate``/``abort`` from an OpenAI-compatible HTTP endpoint.

    Args:
        config (EndpointConfig): where and how to reach the endpoint.

    Example:
        >>> from contextual_drag.inference.endpoint import EndpointConfig
        >>> cfg = EndpointConfig(base_url='http://127.0.0.1:1/v1', model='m')
        >>> engine = EndpointEngine(cfg)
        >>> engine.config.completions_url()
        'http://127.0.0.1:1/v1/completions'
    """

    def __init__(self, config: EndpointConfig) -> None:
        self.config = config
        self._aborted: set[str] = set()

    async def generate(self, prompt: str, sampling_params: Any,
                       request_id: str):
        """
        Yield a single final output for ``prompt``.

        vLLM streams partial outputs and yields many times; a completions
        endpoint returns once. Callers only keep the last value and check
        ``.finished``, so yielding once is equivalent -- but it does mean
        an abort cannot interrupt a request already in flight. It is
        recorded and the result discarded instead.
        """
        if request_id in self._aborted:
            self._aborted.discard(request_id)
            return

        body = {
            'model': self.config.model,
            'prompt': prompt,
            **as_request_kwargs(sampling_params),
        }
        payload = await asyncio.to_thread(self._post, body)

        if request_id in self._aborted:
            self._aborted.discard(request_id)
            return

        choices = payload.get('choices') or []
        outputs = [
            _Completion(
                index=choice.get('index', i),
                text=choice.get('text', ''),
                finish_reason=choice.get('finish_reason'),
            )
            for i, choice in enumerate(choices)
        ]
        yield _RequestOutput(outputs=outputs)

    async def abort(self, request_id: str) -> None:
        """Record an abort. In-flight HTTP requests are not cancellable."""
        self._aborted.add(request_id)

    # -- transport --------------------------------------------------------

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'

        last_error: Optional[Exception] = None
        for attempt in range(max(1, self.config.max_retries)):
            request = urllib.request.Request(
                self.config.completions_url(), data=data, headers=headers)
            try:
                with urllib.request.urlopen(
                        request, timeout=self.config.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as ex:
                detail = ex.read().decode('utf-8', 'replace')[:400]
                # 4xx is a request the endpoint will never accept; retrying
                # just multiplies the same error.
                if 400 <= ex.code < 500:
                    raise RuntimeError(
                        f'{self.config.completions_url()} rejected the '
                        f'request ({ex.code}): {detail}'
                    ) from ex
                last_error = RuntimeError(f'HTTP {ex.code}: {detail}')
            except urllib.error.URLError as ex:
                last_error = RuntimeError(
                    f'cannot reach {self.config.completions_url()}: '
                    f'{ex.reason}'
                )
            if attempt + 1 < self.config.max_retries:
                import time
                time.sleep(2.0 ** attempt)
        raise RuntimeError(
            f'endpoint request failed after {self.config.max_retries} '
            f'attempts: {last_error}'
        )
