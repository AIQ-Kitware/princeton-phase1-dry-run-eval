# Running the evaluation card in a container

## How Kitware runs this card

`python -m magnet.evaluation_new` reads the card's `kwdagger:` block, which
names `cards.pipelines.drag_pipeline()`, and turns it into the seven-node DAG
(`init_inference -> eval_init -> postprocess -> aggregate -> twof_inference
-> eval_twof -> drag_summary`). Each node runs as one `docker run` of the
image built from this repo's `Dockerfile`. The checkout is bind-mounted into
the container at its own absolute path, the node's working directory is
that path, and `PYTHONPATH` is forwarded (both `$REPO` and `$REPO/src`, so
`cards` and `contextual_drag` both resolve from the mount). Results land
under `--output_path`. The DAG backend is tmux on a workstation and Slurm on
the cluster; the card and the image are the same in both cases.

The two inference nodes generate through an OpenAI-compatible endpoint.
Kitware leases that endpoint per node (see Leasing); no vLLM engine is built
inside the node container.

## Build

```bash
cd $REPO
docker build -t contextual-drag-gpu .
```

`MAGNET_REF` in the Dockerfile is the aiq-magnet commit Kitware evaluates
against. To build against the public main instead:

```bash
docker build --build-arg MAGNET_REF=main -t contextual-drag-gpu .
```

The image sets no HuggingFace cache path. The tokenizer for the card's
model is read from the `HF_HOME` the evaluator forwards and mounts, or
fetched from the hub when the container has network access.

## Reproduce the June dry run

On the host you need docker, tmux, infer-stack, and the same aiq-magnet pin
the image carries:

```bash
pip install "aiq-magnet[optional] @ git+https://github.com/AIQ-Kitware/aiq-magnet@5c92d9fc180e1d5deb1c5ec7cd8dc3a64e328e13"
export PYTHONPATH=$REPO:$REPO/src
export HF_HOME=$HOME/.cache/huggingface
```

The June dry run evaluated `Qwen3_8B_NoThinking` on the full GPQA set at
200 questions with n=8. The card's own matrix is the 8-question smoke slice,
so those two values arrive as overrides:

```bash
cd $REPO
python -m magnet.evaluation_new cards/contextual_drag_kwdagger.yaml \
    --output_path runs/contextual_drag \
    --backend tmux \
    --container_image contextual-drag-gpu \
    --container_mounts "$REPO:$HF_HOME" \
    --container_forward_env CONTEXTUAL_DRAG_ENDPOINT,CONTEXTUAL_DRAG_ENDPOINT_MODEL,CONTEXTUAL_DRAG_ENDPOINT_API_KEY,HF_HOME \
    --per_node_leasing \
    --params "matrix: {init_inference.data_fpath: 'data/full_data/gpqa/gpqa.ds', init_inference.max_questions: 200}"
```

Expect `VERIFIED`: drag at or above the 0.05 threshold on the problems that
survive aggregation. The verdict is written to
`runs/contextual_drag/<hash>_<stamp>/verdict.json`, with a `latest` symlink
beside it. Generations are cached per node, so a second run with the same
inputs reuses both inference rounds.

## Leasing

Each inference node's command is wrapped as

```
infer-stack run --endpoint Qwen/Qwen3-8B --ttl 8h --timeout 1800 --queue -- <node command>
```

The alias is the `model_name` of the card's model config
(`Qwen3_8B_NoThinking` names `Qwen/Qwen3-8B`), because that string is what
the REST engine posts as `model`. infer-stack starts the endpoint if it is
not already up, exports `OPENAI_BASE_URL` into the container, and releases
the lease when the node ends. The clean round and the 2F round are separate
jobs and lease separately.

Two catalog settings matter:

- `protocol: completions`, because `contextual_drag` renders prompts through
  the chat template itself and posts them raw; a chat endpoint would apply
  the template twice.
- `reclaim: keep-warm`, because the two rounds lease separately and a `stop`
  policy would reload the weights between them.

Registering the alias, with those two settings, is one `infer-stack catalog
endpoint add` on the evaluating host. Everything else about the endpoint
(engine image, context length, memory fraction) is the evaluator's choice
and does not touch the card.

## What Kitware changes when evaluating

Our runner supplies the host-specific values: which GPU the endpoint lands
on, the HuggingFace cache mount, whether the backend is tmux or Slurm, and
a provenance record beside the verdict. The card, this image and the
command shape above are what we run.
