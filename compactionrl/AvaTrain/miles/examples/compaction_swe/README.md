# CompactionRL SWE rollout

This directory is opt-in. The regular `miles/examples/agentic_swe` rollout and
the normal SWE PPO launcher are not modified.

Each accepted rollout returns one or more execution/summary segments. A summary is
sampled from the same SGLang policy, receives the final SWE reward, and is
included in the PPO loss. The copied summary in the resumed context is masked
out; only the generated summary tokens are trainable.

The working-window defaults follow the paper's GLM-4.7-Flash setup while the
per-call output remains conservative for the 8-GPU Qwen3.5-4B experiment:

```text
working window       65536 tokens
compaction trigger   10240-token reserve (trigger at 55296)
global trajectory    disabled
execution/summary    at most 2048 new tokens
maximum compactions  3
maximum windows      4
recent atomic steps  2
```

The 8-GPU PPO launcher uses `TP=2, CP=2, DP=1` independently for the
four-GPU actor and four-GPU critic. Context parallelism is needed because
dynamic micro-batching can separate different samples but cannot split one
64K segment. `TRAIN_CP_SIZE=1` remains available for short-context diagnostics,
but it is not the supported 64K training layout.

System prompt, original user instruction, assistant actions, and tool
observations all occupy the current 64K working window. There is no independent
16K cap across execution and summary segments. After three compactions, reaching
64K in the fourth window ends the episode; Qwen3.5-4B's native 262144-token
context remains only a final model compatibility check.

Reliability behavior:

- Common SGLang offload, qwen-code timeout/retry, proxy shutdown, sandbox
  cleanup, and empty-trajectory behavior is kept identical to the runnable
  `examples/agentic_swe` baseline.
- A long tool observation is truncated only in the temporary summary prompt;
  the live message history and token ledger are unchanged.
- Reconstructed context tries `recent_steps=2`, then 1, then 0 using the real
  tokenizer length.
- Reaching 64K in the fourth working window returns `finish_reason=stop` and
  prevents every later SGLang generation for that episode.
- Flattened segments are validated and aligned by dropping complete
  trajectories only. Ordinary non-compaction rollout conversion is unchanged.

Auxiliary qwen-code requests are helper branches, not CompactionRL actions. The
default `COMPACTION_AUXILIARY_MODE=serve_untracked` serves them from the rollout
model but keeps their tokens out of the ordered ledger, loss mask, and
credit-assignment chain. For a strict CompactionRL measurement, set
`COMPACTION_AUXILIARY_MODE=reject`; an episode containing any auxiliary request
is then discarded before grading and PPO. Auxiliary tokens must never be
concatenated into the main trajectory, because their position relative to the
canonical agent actions is not defined.

Targeted tests live in `tests/test_compaction_core.py`,
`tests/test_compaction_reliability.py`, and
`tests/test_rollout_data_conversion.py`. GPU/sandbox smoke is still required
before a long run; unit tests do not validate the external sandbox service or
colocated SGLang/Megatron memory behavior.

The PPO launcher saves actor/critic checkpoints every 10 training steps by
default. The final training step is also saved even when it is not a multiple
of 10.
