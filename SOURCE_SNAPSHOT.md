# Source snapshot manifest

This repository is a source-only snapshot of two local training worktrees.
Nested Git submodules were flattened so their local modifications are included.
Runtime data, logs, model artifacts, core dumps, local credentials, `.git` metadata,
`.vendor`, and the backup archives beside the CompactionRL worktree were excluded.

## compactionrl/AvaTrain (outer AvaTrain)

- Source: `/inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/compactionrl/AvaTrain`
- Origin: `https://github.com/sii-avalanche/AvaTrain.git`
- Base commit: `52dc2c71d7c6e81b7c05714a3a2e3ed186ff5a8e`

Local working-tree changes included in this snapshot:

```text
 M .gitignore
 M .gitmodules
 m Megatron-LM
 m miles
?? .vendor/mbridge/
?? PPO_8GPU_DEBUG.md
?? SANDBOX_CAPACITY_FEEDBACK.md
?? scripts/agentic_swe/audit_swe_dev_registry.py
?? scripts/agentic_swe/build_swe_dev_secondary_1000.sh
?? scripts/agentic_swe/finish_swe_dev_after_public.sh
?? scripts/agentic_swe/finish_swe_dev_templates.sh
?? scripts/agentic_swe/finish_verified_templates.sh
?? scripts/agentic_swe/prepare_protocol_bundle.sh
?? scripts/agentic_swe/prepare_swe_dev_dual_project.sh
?? scripts/agentic_swe/prepare_swe_dev_inspire.sh
?? scripts/agentic_swe/prepare_swe_dev_rebuilt_inspire.sh
?? scripts/agentic_swe/prepare_swe_dev_secondary_data.sh
?? scripts/agentic_swe/prepare_verified_inspire.sh
?? scripts/agentic_swe/start_swe_dev_secondary_build.sh
?? scripts/agentic_swe/supervise_swe_dev_secondary_build.sh
?? scripts/agentic_swe/swe_dev_inspire_status.py
?? scripts/agentic_swe/watch_verified_build.sh
?? scripts/agentic_swe_4b/generate.py
?? scripts/agentic_swe_4b/swe.py
?? scripts/train/agentic_swe_qwen35_4b_4gpu.sh
?? scripts/train/check_exact_cu129_runtime.sh
?? scripts/train/check_miles_image_runtime.sh
?? scripts/train/check_runtime_versions.sh
?? scripts/train/inspect_image_python.sh
?? scripts/train/resume_compactionrl_ppo.sh
?? scripts/train/resume_original_ppo_200.sh
?? scripts/train/resume_original_ppo_250.sh
?? scripts/train/resume_original_ppo_from99_to250.sh
?? scripts/train/run.sh
?? scripts/train/run_compactionrl_ppo.sh
?? scripts/train/run_grpo_recovery_step79.sh
?? scripts/train/run_original_ppo_200.sh
?? scripts/train/run_ppo.sh
?? scripts/train/run_qwen35_4b_swe_dev.sh
```

## compactionrl/AvaTrain/miles

- Source: `/inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/compactionrl/AvaTrain/miles`
- Origin: `https://github.com/fnlp-agentRL/miles.git`
- Base commit: `12ddddd9a4ac0e488319d48dfc5ef3a594b3c2e6`

Local working-tree changes included in this snapshot:

```text
 M examples/agentic_swe/generate.py
 M examples/agentic_swe/proxy.py
 M examples/agentic_swe/sandbox.py
 M examples/agentic_swe/swe.py
 M examples/agentic_swe/trajectory.py
 M miles/backends/megatron_utils/actor.py
 M miles/backends/megatron_utils/model.py
 M miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py
 M miles/backends/sglang_utils/arguments.py
 M miles/backends/sglang_utils/sglang_engine.py
 M miles/backends/training_utils/data.py
 M miles/backends/training_utils/loss.py
 M miles/backends/training_utils/loss_hub/advantages.py
 M miles/backends/training_utils/loss_hub/logit_processors.py
 M miles/backends/training_utils/loss_hub/math_utils.py
 M miles/ray/placement_group.py
 M miles/ray/rollout/rollout_data_conversion.py
 M miles/ray/rollout/train_data_conversion.py
 M miles/rollout/data_source.py
 M miles/rollout/inference_rollout/inference_rollout_common.py
 M miles/rollout/inference_rollout/inference_rollout_train.py
 M miles/utils/arguments.py
 M miles/utils/data.py
 M tests/fast/backends/megatron_utils/test_lora_model_branches.py
 M tests/fast/backends/megatron_utils/test_lora_weight_sync_validation.py
 M tests/fast/rollout/inference_rollout/integration/test_over_sampling.py
 M train.py
?? core.1020002
?? core.1020004
?? core.1020005
?? examples/agentic_swe/prepare_inspire_templates.py
?? examples/agentic_swe/prepare_swe_dev_data.py
?? examples/agentic_swe/prepare_swe_dev_dual_project.py
?? examples/agentic_swe/prepare_swe_dev_rebuilt_templates.py
?? examples/agentic_swe/prepare_verified_data.py
?? examples/agentic_swe/run_qwen35_4b_4gpu.sh
?? examples/agentic_swe/run_qwen35_4b_4gpu_ppo.sh
?? examples/agentic_swe/verify_inspire_grader.py
?? examples/agentic_swe/verify_inspire_tunnel.py
?? examples/compaction_swe/README.md
?? examples/compaction_swe/__init__.py
?? examples/compaction_swe/advantages.py
?? examples/compaction_swe/batching.py
?? examples/compaction_swe/config.py
?? examples/compaction_swe/generate.py
?? examples/compaction_swe/model_config.py
?? examples/compaction_swe/prompts.py
?? examples/compaction_swe/proxy.py
?? examples/compaction_swe/run_qwen35_4b_8gpu_ppo.sh
?? examples/compaction_swe/tests/test_advantages_cp.py
?? examples/compaction_swe/tests/test_compaction_config.py
?? examples/compaction_swe/tests/test_compaction_core.py
?? examples/compaction_swe/tests/test_compaction_reliability.py
?? examples/compaction_swe/tests/test_cp_dtype_contract.py
?? examples/compaction_swe/tests/test_generate_integrity.py
?? examples/compaction_swe/tests/test_model_config.py
?? examples/compaction_swe/tests/test_rollout_data_conversion.py
?? examples/compaction_swe/tests/test_tunnel_readiness.py
?? examples/compaction_swe/trajectory.py
?? tests/fast/backends/megatron_utils/test_mixed_rollout_engine_lifecycle.py
?? tests/fast/backends/sglang_utils/test_offload_barrier.py
?? tests/fast/rollout/inference_rollout/test_agentic_empty_trajectory.py
?? tests/fast/rollout/inference_rollout/test_agentic_generate_integrity.py
?? tests/fast/rollout/inference_rollout/test_agentic_proxy_lifecycle.py
?? tests/fast/rollout/inference_rollout/test_swe_grader_timeout.py
?? tests/fast/rollout/inference_rollout/test_task_cleanup.py
```

## compactionrl/AvaTrain/sglang

- Source: `/inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/compactionrl/AvaTrain/sglang`
- Origin: `https://github.com/fnlp-agentRL/sglang.git`
- Base commit: `a72831a0a2d68a79bc3f8257262b5cba6bc0ce54`

Local working tree was clean.

## compactionrl/AvaTrain/Megatron-LM

- Source: `/inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/compactionrl/AvaTrain/Megatron-LM`
- Origin: `https://github.com/fnlp-agentRL/Megatron-LM.git`
- Base commit: `b6c451dbae34f28e85e3239a4a454f11023c158c`

Local working-tree changes included in this snapshot:

```text
 M megatron/core/dist_checkpointing/dict_utils.py
 M megatron/core/dist_checkpointing/strategies/filesystem_async.py
 M megatron/core/dist_checkpointing/strategies/torch.py
 M megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py
 M megatron/core/optimizer/distrib_optimizer.py
 M megatron/training/checkpointing.py
 M tests/unit_tests/dist_checkpointing/test_optimizer.py
 M tests/unit_tests/test_optimizer_cpu_offloading.py
?? tests/unit_tests/dist_checkpointing/test_dict_utils.py
```

## swe-rl/AvaTrain (outer AvaTrain)

- Source: `/inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/swe-rl/AvaTrain`
- Origin: `https://github.com/sii-avalanche/AvaTrain.git`
- Base commit: `52dc2c71d7c6e81b7c05714a3a2e3ed186ff5a8e`

Local working-tree changes included in this snapshot:

```text
 M .gitignore
 M .gitmodules
 m Megatron-LM
 m miles
?? .vendor/mbridge/
?? PPO_8GPU_DEBUG.md
?? PPO_8GPU_FEEDBACK.md
?? PPO_8GPU_FEEDBACK_FEISHU.txt
?? SWEBENCH_EVALUATION.md
?? scripts/agentic_swe/audit_swe_dev_registry.py
?? scripts/agentic_swe/finish_swe_dev_after_public.sh
?? scripts/agentic_swe/finish_swe_dev_templates.sh
?? scripts/agentic_swe/finish_verified_templates.sh
?? scripts/agentic_swe/prepare_protocol_bundle.sh
?? scripts/agentic_swe/prepare_swe_dev_dual_project.sh
?? scripts/agentic_swe/prepare_swe_dev_inspire.sh
?? scripts/agentic_swe/prepare_swe_dev_rebuilt_inspire.sh
?? scripts/agentic_swe/prepare_swe_dev_secondary_data.sh
?? scripts/agentic_swe/prepare_verified_inspire.sh
?? scripts/agentic_swe/start_swe_dev_secondary_build.sh
?? scripts/agentic_swe/supervise_swe_dev_secondary_build.sh
?? scripts/agentic_swe/swe_dev_inspire_status.py
?? scripts/agentic_swe/watch_verified_build.sh
?? scripts/agentic_swe_4b/generate.py
?? scripts/agentic_swe_4b/swe.py
?? scripts/train/agentic_swe_qwen35_4b_4gpu.sh
?? scripts/train/check_exact_cu129_runtime.sh
?? scripts/train/check_miles_image_runtime.sh
?? scripts/train/check_runtime_versions.sh
?? scripts/train/inspect_image_python.sh
?? scripts/train/resume_original_grpo_200.sh
?? scripts/train/run.sh
?? scripts/train/run_grpo_recovery_step79.sh
?? scripts/train/run_ppo.sh
?? scripts/train/run_qwen35_4b_swe_dev.sh
?? scripts/train/watch_compaction_then_start_ppo.sh
```

## swe-rl/AvaTrain/miles

- Source: `/inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/swe-rl/AvaTrain/miles`
- Origin: `https://github.com/fnlp-agentRL/miles.git`
- Base commit: `12ddddd9a4ac0e488319d48dfc5ef3a594b3c2e6`

Local working-tree changes included in this snapshot:

```text
 M examples/agentic_swe/generate.py
 M examples/agentic_swe/proxy.py
 M examples/agentic_swe/sandbox.py
 M examples/agentic_swe/swe.py
 M examples/agentic_swe/trajectory.py
 M miles/backends/megatron_utils/actor.py
 M miles/backends/megatron_utils/model.py
 M miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py
 M miles/backends/sglang_utils/arguments.py
 M miles/backends/sglang_utils/sglang_engine.py
 M miles/backends/training_utils/loss_hub/logit_processors.py
 M miles/backends/training_utils/loss_hub/math_utils.py
 M miles/ray/placement_group.py
 M miles/rollout/data_source.py
 M miles/rollout/inference_rollout/inference_rollout_common.py
 M miles/rollout/inference_rollout/inference_rollout_train.py
 M miles/utils/arguments.py
 M miles/utils/data.py
 M tests/fast/backends/megatron_utils/test_lora_model_branches.py
 M tests/fast/rollout/inference_rollout/integration/test_over_sampling.py
?? core.1020002
?? core.1020004
?? core.1020005
?? examples/agentic_swe/prepare_inspire_templates.py
?? examples/agentic_swe/prepare_swe_dev_data.py
?? examples/agentic_swe/prepare_swe_dev_dual_project.py
?? examples/agentic_swe/prepare_swe_dev_rebuilt_templates.py
?? examples/agentic_swe/prepare_verified_data.py
?? examples/agentic_swe/run_qwen35_4b_4gpu.sh
?? examples/agentic_swe/run_qwen35_4b_4gpu_ppo.sh
?? examples/agentic_swe/verify_inspire_grader.py
?? examples/agentic_swe/verify_inspire_tunnel.py
?? tests/fast/backends/sglang_utils/test_offload_barrier.py
?? tests/fast/rollout/inference_rollout/test_agentic_empty_trajectory.py
?? tests/fast/rollout/inference_rollout/test_agentic_proxy_lifecycle.py
?? tests/fast/rollout/inference_rollout/test_swe_grader_timeout.py
?? tests/fast/rollout/inference_rollout/test_task_cleanup.py
```

## swe-rl/AvaTrain/sglang

- Source: `/inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/swe-rl/AvaTrain/sglang`
- Origin: `https://github.com/fnlp-agentRL/sglang.git`
- Base commit: `a72831a0a2d68a79bc3f8257262b5cba6bc0ce54`

Local working tree was clean.

## swe-rl/AvaTrain/Megatron-LM

- Source: `/inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/swe-rl/AvaTrain/Megatron-LM`
- Origin: `https://github.com/fnlp-agentRL/Megatron-LM.git`
- Base commit: `b6c451dbae34f28e85e3239a4a454f11023c158c`

Local working-tree changes included in this snapshot:

```text
 M megatron/core/dist_checkpointing/dict_utils.py
 M megatron/core/dist_checkpointing/strategies/filesystem_async.py
 M megatron/core/dist_checkpointing/strategies/torch.py
 M megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py
 M megatron/core/optimizer/distrib_optimizer.py
 M megatron/training/checkpointing.py
 M tests/unit_tests/dist_checkpointing/test_optimizer.py
 M tests/unit_tests/test_optimizer_cpu_offloading.py
?? tests/unit_tests/dist_checkpointing/test_dict_utils.py
```
