# SWE-bench 评测代码与使用说明

本文记录 AvaTrain 中现有 SWE-bench 评测链路的真实能力、代码入口和可直接执行的命令。适用工作目录：

```text
/inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/swe-rl/AvaTrain
```

## 1. 当前实现状态

现有代码已经具备端到端单任务评测能力：

1. 在 Inspire Sandbox 中启动待修改仓库；
2. 运行未修改的 `qwen-code` agent，并通过本机 SGLang policy 生成工具调用；
3. 收集 agent 产生的 Git patch；
4. 立即释放 agent 使用过的可变 sandbox；
5. 新建一个干净 sandbox，重新应用 candidate patch 和数据集提供的 test patch；
6. 执行任务的测试命令并解析 `FAIL_TO_PASS`、`PASS_TO_PASS`；
7. 将 reward、token trajectory、状态等保存为 Miles rollout `.pt` 文件。

目前没有独立、经过 GPU 实跑验证的“输入任意 Megatron checkpoint，一条命令跑完 500 个 SWE-bench Verified 样本并生成官方 leaderboard predictions JSON”的 wrapper。不要把通用 `inference_rollout_eval.py` 或 Terminal-Bench 的 eval launcher 当成这个 wrapper。

## 2. 代码入口

| 文件 | 作用 |
|---|---|
| `miles/examples/agentic_swe/generate.py` | 单个 SWE episode：创建 sandbox、运行 agent、收集 patch、调用 grader、清理 sandbox |
| `miles/examples/agentic_swe/proxy.py` | 将 qwen-code 的 OpenAI 请求代理到 Miles/SGLang，并记录真实采样 token |
| `miles/examples/agentic_swe/trajectory.py` | 构造 token、loss mask、reward 和最终 Miles `Sample` |
| `miles/examples/agentic_swe/swe.py` | 干净 sandbox 中应用 patch、执行测试、解析结果、计算 reward |
| `miles/examples/agentic_swe/sandbox.py` | Inspire Sandbox 创建、重试、反向隧道和清理 |
| `miles/examples/agentic_swe/verify_inspire_grader.py` | 用一个数据样本真实检查 sandbox 和 grader 链路 |
| `miles/examples/agentic_swe/run_qwen35_4b_4gpu.sh` | Qwen3.5-4B 的底层 Miles/SGLang 启动器，支持 `debug-rollout-only` |
| `scripts/train/run.sh` | 已做环境、数据、checkpoint、凭据和 GPU 预检的上层入口 |
| `miles/miles/ray/rollout/debug_data.py` | 将 rollout sample 保存为 `.pt` |
| `miles/miles/utils/debug_utils/display_debug_rollout_data.py` | 查看 `.pt` 中的 reward、状态和样本内容 |
| `miles/tools/convert_torch_dist_to_hf.py` | 将某个 Megatron `iter_XXXXXXX` actor checkpoint 导出为 HF 权重 |

`miles/miles/rollout/inference_rollout/inference_rollout_eval.py` 是 Miles 的通用 eval 数据调度器，本项目当前没有给它配置 SWE-bench 专用的完整批量入口。`miles/examples/eval/` 下现有脚本主要面向数学任务和 Terminal-Bench。

## 3. Reward 与官方 resolved 的区别

当前训练和在线评测共用以下稠密 reward：

```text
F2P_ratio = 通过的 FAIL_TO_PASS 数 / 有效 FAIL_TO_PASS 总数
P2P_ratio = 1 - 失败的 PASS_TO_PASS 数 / 有效 PASS_TO_PASS 总数
reward    = F2P_ratio * P2P_ratio
```

实现位于 `miles/examples/agentic_swe/swe.py` 的 `SweTask.score()`。

该实现使用 SWE-bench 4.1.0 的 repo-specific log parser；SWE-Dev 数据使用项目固定的 pytest parser。它适合训练中的稠密反馈和同一实现下的模型横向比较，但不能直接冒充官方 leaderboard 的二值 `resolved` 指标：

- `reward=1.0` 可作为“本地目标测试全部满足”的近似成功率；
- 平均 reward 是稠密测试通过率，不是官方 `% resolved`；
- 正式对外报告 leaderboard 结果时，仍应保存每题 patch，并交给官方 SWE-bench harness 生成 report。

## 4. 数据与运行时

当前固定路径：

```bash
AVA_ROOT=/inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/swe-rl/AvaTrain
SHARED_ROOT=/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk
HF_BASE=$SHARED_ROOT/model/Qwen/Qwen3.5-4B
REF_LOAD=$SHARED_ROOT/model/Qwen/Qwen3.5-4B_torch_dist
SWEBENCH_RUNTIME_DIR=$SHARED_ROOT/.deps/swebench-runtime-py312
VERIFIED_DATA=$SHARED_ROOT/swe-rl/data/swe_verified_500_avatrain_qwen_code_0.21.0.jsonl
SWE_DEV_DUAL_DATA=$SHARED_ROOT/swe-rl/data/swe_dev_1000_dual_project_avatrain_qwen_code_0.21.0.jsonl
SMOKE_DATA=$SHARED_ROOT/swe-rl/data/smoke_verified_1_avatrain_qwen_code_0.21.0.jsonl
```

数据 JSONL 每行必须包含 `prompt`、`label` 和 `metadata`。grader 至少依赖以下 metadata：

```text
instance_id, repo, repo_workdir, base_commit, inspire_template,
docker_image_default_user, docker_image_env, test_patch,
FAIL_TO_PASS, PASS_TO_PASS, install_config.test_cmd
```

`scripts/train/run.sh` 还会检查：SWE-bench 版本为 4.1.0、Qwen3.5 HF 权重完整、torch_dist release checkpoint 完整、协议 bundle 可执行，以及 Sandbox API 凭据存在。

## 5. 先执行预检

在 Miles GPU 镜像内执行：

```bash
cd /inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/swe-rl/AvaTrain

CUDA_VISIBLE_DEVICES=0 \
RUNTIME_MODE=image \
bash scripts/train/run.sh check
```

成功标志：

```text
READY: GPU runtime, checkpoints, data, protocol, and credentials passed preflight.
```

这里只检查运行时和静态资源，不创建 sandbox，也不进行模型生成。

## 6. 只验证真实 grader

以下命令不运行模型。它取 `SMOKE_DATA` 第一条任务，在 sandbox 中形成空 patch，然后在另一个干净 sandbox 中执行真实测试，用来检查模板、凭据、test patch、测试环境和 parser：

```bash
cd /inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/swe-rl/AvaTrain

set -a
source scripts/train/.env
set +a

export SWEBENCH_RUNTIME_DIR=/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/.deps/swebench-runtime-py312
export PYTHONPATH="$PWD/Megatron-LM:$PWD/miles:$PWD/miles/examples/agentic_swe:/inspire/hdd/project/exploration-topic/public/sywang/fzk/.deps/inspire-sandbox${PYTHONPATH:+:$PYTHONPATH}"

/usr/bin/python miles/examples/agentic_swe/verify_inspire_grader.py \
  --data /inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/swe-rl/data/smoke_verified_1_avatrain_qwen_code_0.21.0.jsonl
```

成功标志：

```text
READY official SWE-bench grader instance=... empty_patch_reward=...
```

空 patch 得到 0 reward 通常是正常结果；该命令验证的是 grader 能完整结束，而不是要求空 patch 通过测试。

## 7. 运行一次端到端模型评测

以下命令使用 1 张 GPU、1 个 Verified 样本，真实运行 qwen-code、生成 patch，并在干净 sandbox 中评分：

```bash
cd /inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/swe-rl/AvaTrain

CUDA_VISIBLE_DEVICES=0 \
RUNTIME_MODE=image \
TRAIN_DATASET=verified \
bash scripts/train/run.sh rollout-1gpu
```

默认结果文件：

```text
/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/swe-rl/smoke/verified_rollout_0.pt
```

成功标志：

```text
READY: one real Agentic SWE rollout completed.
```

重要：`rollout-1gpu` 会开启 Miles 的 `--debug-rollout-only`。这个模式直接由 SGLang 加载 `HF_CHECKPOINT`，不会初始化 Megatron actor，也不会通过 `LOAD_CHECKPOINT` 更新 SGLang 权重。因此：

- 默认命令评测的是基础 `Qwen3.5-4B` HF 模型；
- 仅设置 `LOAD_CHECKPOINT=/path/to/actor` 并不能评测训练后的 actor；
- 评测训练 actor 时，必须先将目标 iteration 导出为 HF，再把 `HF_CHECKPOINT` 指向导出目录。

## 8. 导出并评测训练后的 actor

先确认目标 actor iteration 完整：

```bash
ACTOR_CKPT=/path/to/ppo_actor_checkpoint
STEP=124
ITER_DIR=$(printf '%s/iter_%07d' "$ACTOR_CKPT" "$STEP")

test -s "$ITER_DIR/.metadata"
test -s "$ITER_DIR/common.pt"
```

只有 actor checkpoint 可以用于生成；critic checkpoint 不能替代 actor。然后在 GPU 镜像中导出：

```bash
cd /inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/swe-rl/AvaTrain

ACTOR_CKPT=/path/to/ppo_actor_checkpoint
STEP=124
ITER_DIR=$(printf '%s/iter_%07d' "$ACTOR_CKPT" "$STEP")
HF_EXPORT=/inspire/hdd/global_user/wangsiyin-240108120103/fzk/model/eval_hf/qwen35_4b_ppo_step124

export PYTHONPATH="$PWD/Megatron-LM:$PWD/miles:$PWD/.vendor/mbridge${PYTHONPATH:+:$PYTHONPATH}"

/usr/bin/python miles/tools/convert_torch_dist_to_hf.py \
  --input-dir "$ITER_DIR" \
  --output-dir "$HF_EXPORT" \
  --origin-hf-dir /inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/model/Qwen/Qwen3.5-4B \
  --vocab-size 248320

test -s "$HF_EXPORT/model.safetensors.index.json"
```

输出目录已存在时，转换器默认拒绝覆盖。只有在确认旧导出不需要保留时才添加 `--force`。

用导出的训练 actor 做单样本端到端评测：

```bash
cd /inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/swe-rl/AvaTrain

CUDA_VISIBLE_DEVICES=0 \
RUNTIME_MODE=image \
TRAIN_DATASET=verified \
HF_CHECKPOINT=/inspire/hdd/global_user/wangsiyin-240108120103/fzk/model/eval_hf/qwen35_4b_ppo_step124 \
REF_LOAD=/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/model/Qwen/Qwen3.5-4B_torch_dist \
DEBUG_ROLLOUT_DATA='/inspire/hdd/global_user/wangsiyin-240108120103/fzk/model/eval_results/ppo_step124_{rollout_id}.pt' \
bash scripts/train/run.sh rollout-1gpu
```

这里的 `REF_LOAD` 仍由上层预检要求存在，但 `debug-rollout-only` 实际生成权重来自 `HF_CHECKPOINT`。

## 9. 查看评测结果

查看 reward 和汇总指标，不打印完整 token/sample：

```bash
cd /inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/swe-rl/AvaTrain

PYTHONPATH="$PWD/miles${PYTHONPATH:+:$PYTHONPATH}" \
/usr/bin/python -m miles.utils.debug_utils.display_debug_rollout_data \
  --load-debug-rollout-data '/inspire/hdd/global_user/wangsiyin-240108120103/fzk/model/eval_results/ppo_step124_{rollout_id}.pt' \
  --category train \
  --no-show-samples
```

查看单条样本时去掉 `--no-show-samples`。`.pt` 内主要字段包括：

```text
label, reward, status, response_length, prompt, response, metadata
```

日志中的关键成功记录：

```text
Agent finished label=... exit_code=...
SWE grader instance=... reward=...
Trajectory finalized label=... response_tokens=... status=... reward=...
```

`status=truncated` 表示 trajectory 达到 token 上限；agent shell 超时、sandbox 创建失败和 grader 基础设施失败应作为失败任务单独排查，不能静默计成普通 0 reward。

## 10. Sandbox 并发与清理

双项目数据使用：

```text
SANDBOX_CONCURRENCY_PRIMARY
SANDBOX_CONCURRENCY_SECONDARY
```

并发限制覆盖完整 episode，而不只是 create API。`generate.py` 在进入 agent episode 前取得对应项目 semaphore；agent sandbox 收集 patch 后立即 kill，之后才创建 grader sandbox，因此正常峰值不会同时为每个样本保留两个 sandbox。

评测前只应清理确认属于本项目且已失去活跃日志保护的 stale sandbox：

```bash
cd /inspire/hdd/project/exploration-topic/public/sywang/fzk

bash tools/sandbox_cleanup/cleanup_stale_swe_sandboxes.sh \
  --log-dir /inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/logs/swe-rl
```

先检查 dry-run 输出；确认候选无误后再加 `--apply`。不要删除脚本标为 `PROTECTED` 的 sandbox。

## 11. 当前不应使用的做法

1. 不要用 `train-4gpu` 代替评测，它会执行优化器更新并保存训练 checkpoint。
2. 不要认为 `LOAD_CHECKPOINT` 会让 `debug-rollout-only` 自动加载训练 actor。
3. 不要把平均稠密 reward 直接写成官方 `% resolved`。
4. 不要把 `verify_inspire_grader.py` 的空 patch 分数当作模型能力结果。
5. 不要在评测过程中运行无保护的全量 sandbox 删除命令。
6. 不要使用 critic checkpoint 做推理或 patch 生成。

## 12. 尚缺的正式批量评测功能

若要形成可对外报告的完整 SWE-bench 评测，还需要单独实现并 GPU 验证以下闭环：

1. 从目标 actor iteration 稳定导出 HF；
2. 按数据集顺序对 500 个 Verified 样本各生成一次，支持断点续评和失败重试；
3. 为每个 `instance_id` 单独落盘 candidate patch，而不仅是序列化 token sample；
4. 生成 SWE-bench predictions JSONL；
5. 使用官方 harness 生成 resolved report；
6. 将基础设施失败、超时、截断、空 patch 和真实未解决样本分开统计；
7. 固定 temperature、seed、agent/qwen-code 版本、token budget 和 sandbox template manifest。

在该批量入口完成前，本文第 5 至第 9 节适合运行时门禁、单样本回归和本地稠密 reward 对比，不应宣称为完整 leaderboard 评测。
