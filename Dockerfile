# syntax=docker/dockerfile:1.7
# contextual_drag evaluation image: the environment the Contextual Drag cards
# run in.
#
# Kitware evaluates a card by turning its `kwdagger:` block into a DAG and
# running every node of that DAG as one `docker run` of this image, with the
# checkout bind-mounted at its own absolute path and the node's cwd there. The
# copy of this repo baked below supplies the installed environment; the mount
# supplies the code that actually runs. See docs/containerized_evaluation.md.
#
#   docker build -t contextual-drag-gpu .
#
# The inference nodes generate through an OpenAI-compatible endpoint that is
# leased per node (CONTEXTUAL_DRAG_ENDPOINT / OPENAI_BASE_URL). vLLM is still
# installed so the in-process engine path keeps working when no endpoint is
# given, which is what makes this a GPU image.
#
# No HF_HOME or TRANSFORMERS_CACHE is set here on purpose: the evaluator
# forwards HF_HOME and mounts the host's HuggingFace cache, and the container
# runs as the invoking user, so a cache path baked into the image would be
# unwritable.
#
# MAGNET_REF is the aiq-magnet commit Kitware evaluates against. It is
# on AIQ-Kitware/aiq-magnet main (the kwdagger execution merge, PR #94);
# `--build-arg MAGNET_REF=main` builds against the tip of main instead.
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}
ARG MAGNET_REF=5c92d9fc180e1d5deb1c5ec7cd8dc3a64e328e13

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        git \
        jq \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade uv

WORKDIR /opt/src

# MAGNET first, so the heavy layer is stable across edits to this repo. No
# extras: the nodes read no HELM output, and the lease that supplies the
# endpoint is acquired outside the container by the evaluator's magnet.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        "aiq-magnet @ git+https://github.com/AIQ-Kitware/aiq-magnet@${MAGNET_REF}" \
        'safer>=2.0'

# Runtime dependencies in a source-independent layer: pyproject's base plus
# the inference, eval and analysis extras.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        'datasets>=3.1,<4' \
        'joblib>=1.3' \
        'numpy>=1.26' \
        'pandas>=2' \
        'scriptconfig>=0.8.2' \
        'sympy>=1.12' \
        'tqdm>=4.66' \
        'accelerate>=1.10,<2' \
        'torch>=2.8,<2.9' \
        'transformers>=4.55,<5' \
        'vllm>=0.10,<0.11' \
        'math-verify>=0.8.0' \
        'zss>=1.2.0' \
        'python-Levenshtein>=0.27' \
        'networkx>=3' \
        'matplotlib>=3.8'

# This repo, without dependencies so ordinary edits only invalidate this
# small layer and the magnet used is the one pinned above, not the copy under
# submodules/.
COPY . /opt/src/princeton-phase1-dry-run-eval
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps -e /opt/src/princeton-phase1-dry-run-eval

WORKDIR /opt/src/princeton-phase1-dry-run-eval
CMD ["bash"]
