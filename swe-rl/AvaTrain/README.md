# AvaTrain

## Installation

Run the following command to configure git submodules and fetch & build dependencies. Note `causal-conv1d` compilation may fail in resource-contrained node.

```bash
make init
```

## Development

Custom training scripts should be added/modified in `scripts` folder, and committed to `AvaTrain` repository. Use `uv run scripts/custom-script.py` to launch any training script. Modification of underlying frameworks (`megatron-core`, `miles` and `sglang`) should be committed/pushed to their own repositories.

## Agentic SWE RL

After preparing a task JSONL file and model checkpoints, run the agentic SWE
example from the workspace root:

```bash
export HF_CHECKPOINT=/path/to/Qwen3.5-35B-A3B-sft
export REF_LOAD=/path/to/Qwen3.5-35B-A3B-sft_torch_dist
export SBX_API_KEY=...
export SBX_API_URL=https://qz-sbx-api.sii.edu.cn
bash scripts/train/agentic_swe.sh /path/to/swe_verified_train.jsonl
```

The wrapper installs the locked `swe` extra and checks that `wstunnel` is
available on `PATH`. See [`miles/examples/agentic_swe/README.md`](miles/examples/agentic_swe/README.md) for the task metadata contract.
