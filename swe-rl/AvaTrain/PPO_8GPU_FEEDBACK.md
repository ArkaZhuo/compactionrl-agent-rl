# Qwen3.5-4B PPO 使用反馈与修复说明

当前更新：2026-08-08 UTC


我们在Avatrain 的基础上接入 Qwen3.5-4B，使用 8 张 H100 跑了
SWE-Dev 上的 PPO。实际训练布局是 actor 4 卡、critic 4 卡，训练侧保持 `TP=2`，
rollout 侧则在 8 张卡上各放一个 TP=1 的 SGLang engine。

整体上，代码能够完成 rollout、critic/actor 更新、权重同步和 checkpoint 保存，说明
PPO 主流程是可用的。不过连续训练和任务重启是两种不同的压力场景。跑得久一些，我们遇到了几个稳定性问题。这些问题大多不在
PPO loss 本身，而是在 optimizer checkpoint、agent 请求生命周期和分布式写盘这些边界上。

下面只记录我们认为属于原代码实现或默认恢复路径的问题。

## 总体结论

这套代码跑短任务比较顺利，但如果目标是长时间 PPO 训练，我们发现需要重点补强以下几处：

1. `dp_reshardable` checkpoint 不能把 Adam 的标量 `step` 当作参数 tensor 分片。
2. agent episode 结束不等于底层 SGLang 请求已经退出，offload 前需要真正的排空屏障。
3. sandbox 不仅要在正常路径调用 `kill()`，还要提前释放 agent sandbox，并覆盖异常、取消和
   surplus rollout 的清理路径。
4. 共享文件系统上的异步 checkpoint 需要原子落盘和有限重试。
5. PPO 的 actor、critic 和 rollout dataset state 应当作为一个恢复点校验，不能只看单个 tracker。


## 1. Adam `step` 在 8 卡 checkpoint 中变成了向量

### 现象

从 4 卡训练得到的 checkpoint 恢复到 8 卡后，同一个进程可以继续训练，也可以保存新的
checkpoint。但任务退出后，再从这个 8 卡 checkpoint 启动，actor 第一次执行 Adam 更新时
会报：

```text
RuntimeError: a Tensor with 32 elements cannot be converted to Scalar
```

这个现象一开始比较容易误判。因为 checkpoint 加载本身会显示成功，critic 也可能先完成
更新，错误只在 actor 进入 PyTorch fused Adam 时出现。它和 rollout、reward、sandbox 或
SGLang 没有关系。

### 原因

Adam 的状态大致是：

```python
optimizer.state[param] = {
    "step": scalar_tensor,
    "exp_avg": parameter_tensor,
    "exp_avg_sq": parameter_tensor,
    "master_param": parameter_tensor,
}
```

`exp_avg`、`exp_avg_sq` 和 `master_param` 与参数同形状，适合进入
`dp_reshardable` 的 per-parameter bucket。`step` 不一样，它是 optimizer 的标量计数，
正常情况下应该通过 `param_groups` 保存和恢复。

原实现从 optimizer state 取出所有 tensor 后，没有排除 `step`。当 DP bucket 需要补齐
32 个元素时，padding 逻辑也为 `step` 创建了一个 32 元素 tensor。加载 checkpoint 时，
`param_groups` 中正确的标量 `step` 先被恢复，随后又被 per-parameter state 中错误的向量
覆盖。直到 fused Adam 尝试把它读取为标量时，问题才真正暴露。

这也解释了为什么同一个 8 卡进程可以从 step 20 一直训练到后面：保存 checkpoint 不会把
磁盘上的错误状态重新加载到当前 optimizer。只有任务重启后才会触发。

### 修改

修改文件：

```text
Megatron-LM/megatron/core/optimizer/distrib_optimizer.py
```

保存 `dp_reshardable` parameter state 时移除 `step`：

```python
tensors = self._get_main_param_and_optimizer_states(model_param)
tensors.pop("step", None)
```

加载旧 checkpoint 时也过滤 `step`：

```python
src_tensors = {key: value for key, value in src_tensors.items() if key != "step"}
self._set_main_param_and_optimizer_states(model_param, src_tensors)
```

保存端修改保证新 checkpoint 不再产生错误状态；加载端修改用于兼容已经保存的旧
checkpoint。这里没有丢弃 Adam moments，也没有把 optimizer 重置为零。正确的标量
`step` 已经从 `param_groups` 恢复，`exp_avg`、`exp_avg_sq`、master weights 和模型权重
仍然按原逻辑严格加载。

### 验证

我们做了两层验证：

- 增加两个单元测试，分别覆盖新 checkpoint 保存和旧 checkpoint 兼容加载。
- 从包含错误 `step` 的 iteration 20 恢复，成功生成 iteration 21；随后启动新任务，
  从 iteration 21 再次恢复并完成后续 actor/critic 更新。

这个问题可以确认已经修复，不需要为了兼容旧 checkpoint 回退训练进度。

## 2. Rollout 已结束，但 SGLang 请求还没有真正退出

### 现象

长跑到 rollout 边界时，我们看到 64 条 trajectory 已经全部完成，随后开始释放 SGLang
显存。但同一时间仍有晚到的 `/generate`、prefill 和 decode。最终多个 SGLang engine 在
请求池分配处报：

```text
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

崩溃窗口里还伴随这些信号：

```text
BrokenPipeError
CancelledError
Cache not flushed because there are pending requests
```

后面的 `503 no_available_workers` 和 `connection refused` 是 engine 已经退出后的连锁反应，
不是最初原因。这次故障窗口中没有 `cuMemCreate` 或 PyTorch OOM，因此不能归因于样本太长
或显存比例过高。

### 原因

SWE agent 的一次 episode 不是一次简单的模型调用。它包含 Qwen CLI、反向隧道、本地
proxy、多轮工具调用和多次 SGLang `/generate`。顶层 episode 返回，只说明上层任务结束；
如果 CLI 或隧道先断开，proxy handler 可能收到 `BrokenPipeError`，但已经提交到 SGLang
event loop 的 HTTP coroutine 仍可能处在 prefill/decode。

原来的 `ModelProxy.close()` 主要关闭 HTTP server 和 Python future，没有同时等待实际的
HTTP coroutine 退出。Miles 在 offload 前调用的 `abort_request(abort_all=True)` 也是异步
通知：HTTP 200 只表示消息已发送到 scheduler，并不保证 scheduler、KV cache 和
Hybrid/Mamba request pool 已经排空。

因此旧流程存在下面的竞态：

```text
顶层 rollout 完成
  -> proxy future 被取消
  -> abort 消息发出
  -> flush/reset/offload 开始
  -> 晚到的 prefill 继续访问已经重置的请求池
```

这是时序问题，所以前面几十步都正常并不能证明代码没有问题。降低并发只会降低撞到这个
窗口的概率，不会从语义上解决它。

### 修改一：让 proxy 真正 drain

修改文件：

```text
miles/examples/agentic_swe/proxy.py
```

我们给 `ModelProxy` 增加了一个共享的 `threading.Condition`，统一管理：

- `_closing`：关闭开始后拒绝新请求；
- `_active_completions`：仍在 handler 中执行的 completion；
- `_inflight`：提交到 rollout event loop 的 future；
- `_pending_generations`：尚未执行完 `finally` 的真实 HTTP coroutine。

关闭过程改成：先原子设置 `_closing=True`，停止接收新连接，取消已登记的 future，然后等待
active handler、future 和 coroutine 全部归零。只有 drain 成功后才关闭 server。如果在规定
时间内不能排空，就明确抛出 `TimeoutError`，而不是带着未知的模型请求继续 offload。

特别需要等待 `_pending_generations`。跨线程 future 在调用 `cancel()` 后可能很快显示为
cancelled，但 event loop 中对应 coroutine 的清理并不一定已经完成，只看 future 不足以构成
生命周期屏障。

### 修改二：offload 前使用 SGLang 的暂停屏障

修改文件：

```text
miles/miles/backends/sglang_utils/sglang_engine.py
```

旧顺序是：

```python
abort_all_requests()
flush_cache()
release_memory_occupation()
```

修改后是：

```python
pause_generation(mode="abort", timeout_seconds=120)
flush_cache(timeout_seconds=60, retry_interval_seconds=1)
release_memory_occupation()
```

`pause_generation(mode="abort")` 会先关闭新请求准入，abort 现有请求，并等待 SGLang 的
model-update lock 释放。它返回后再执行 `flush_cache()`，才有明确的先后关系。pause 或
flush 超时都会阻止 offload 继续执行，避免请求池仍在使用时释放显存。

### 验证状态

新增测试：

```text
miles/tests/fast/rollout/inference_rollout/test_agentic_proxy_lifecycle.py
miles/tests/fast/backends/sglang_utils/test_offload_barrier.py
```

测试覆盖了关闭后禁止新请求、取消并等待实际 generation、状态归零，以及
`pause(abort) -> flush -> release` 的调用顺序。

代码和定向测试已经通过，但这个问题依赖真实的 SGLang scheduler 时序。我们建议在宣布
完全修复前，从现有 iteration 74 只跑一个 step 75，确认 8 个 engine 都能完成 pause、flush、
release、权重恢复和下一轮生成。

## 3. Sandbox 清理时机和取消传播不完整

### 原实现并非没有清理，但清理得太晚

原代码已经使用 `AsyncExitStack` 注册了 `runtime.kill()`，grader 使用的干净 sandbox 也有
`finally: await clean.kill()`。所以这里的问题不是“正常结束后从来不删除 sandbox”，而是
清理时机和异常传播不够完整。

原来的 episode 顺序是：

```text
创建 agent sandbox
  -> agent 修改仓库
  -> 在 agent sandbox 仍存活时创建 clean grader sandbox
  -> grader 完成
  -> 删除 clean grader sandbox
  -> episode 退出时才删除 agent sandbox
```

也就是说，在 grader 运行期间，同一条 trajectory 会同时占用两个 sandbox。PPO 一批有
64 条 trajectory，这个重叠会明显放大平台容量压力。更重要的是，一旦外层 rollout task
被取消或组内其他 task 先报错，清理能否执行取决于取消有没有真正传播并等待到 episode 的
`finally`。

### 问题一：组内异常不会自动清理其他 sibling

GRPO/PPO 的同组采样通过 `asyncio.gather()` 等待。如果组内某个 agent 因网络、sandbox
或解析异常提前失败，`gather()` 会把第一个异常抛给调用方，但其他 task 可能还在后台运行。
对于普通纯 Python task，这可能只是多执行了一会儿；对于 agentic SWE task，每个 sibling
都持有真实 sandbox、隧道和 grader，后台继续运行会占用 sandbox 容量，也会让 rollout 已经
结束后仍有模型请求进入 SGLang。

修改文件：

```text
miles/miles/rollout/inference_rollout/inference_rollout_common.py
```

对 `asyncio.gather()` 增加异常清理：

```python
try:
    group = await asyncio.gather(*tasks)
except BaseException:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    raise
```

这里第二次 `gather(..., return_exceptions=True)` 很重要。只调用 `cancel()` 不代表 task 的
`finally` 已经执行；等待所有 sibling 完成 unwind，sandbox 删除、proxy 关闭和隧道清理才
真正结束。

### 问题二：达到目标 batch 后仍等待 surplus rollout

原来的 `abort()` 会向 SGLang 发送 abort，但对于 non-partial rollout，仍通过
`as_completed_async(pendings)` 等待所有多余 task 自然结束。一个 surplus task 如果正卡在
agent 命令或 grader，就会继续占用 sandbox，并把已经收齐的整个 batch 拖住。

修改文件：

```text
miles/miles/rollout/inference_rollout/inference_rollout_train.py
```

对于 non-partial rollout，收齐目标 batch 后直接取消 pending group，并等待它们完成清理：

```python
for task in pendings:
    task.cancel()
await asyncio.gather(*pendings, return_exceptions=True)
```

partial rollout 仍保留原来的语义，因为它需要收集未完成响应供下一轮继续使用。

同时，无动态过滤时不再固定按 `over_sampling_batch_size` 超额创建任务，而是只按当前缺口
补齐：

```python
request_size = min(args.over_sampling_batch_size, missing_groups)
```

这不是依赖事后清理，而是从源头避免创建本批根本用不到的 agent sandbox。

### 问题三：agent sandbox 保留到了 grader 结束

我们把生成过程拆成“收集 patch”和“独立评分”两个阶段。修改文件：

```text
miles/examples/agentic_swe/generate.py
miles/examples/agentic_swe/swe.py
```

现在的顺序是：

```python
runtime = await sandbox.create_sandbox(...)
try:
    async with sandbox.reverse_tunnel(...):
        await task.setup(runtime)
        await sandbox.run(runtime, ...)
        patch = await task.collect_patch(runtime)
finally:
    await runtime.kill()

reward = await task.grade_patch(patch)
```

拿到 patch 后，agent sandbox 已经没有继续保留的必要，因此在创建 grader sandbox 前就通过
`finally` 删除。无论 setup、agent 命令、patch 收集还是外层取消在哪一步失败，这个
`runtime.kill()` 都会执行。

grader 仍然使用全新的 sandbox，保持评测隔离，并继续用独立的 `finally` 清理：

```python
clean = await sandbox.create_sandbox(...)
try:
    return await run_grader(clean)
finally:
    await clean.kill()
```

grader 超时会记为 reward 0，但返回发生在 `try` 内，仍会经过 `finally` 删除 clean sandbox。
因此超时样本不会一直占用平台资源直到三小时 TTL 到期。

### 清理顺序

一条正常 trajectory 现在按以下顺序释放资源：

```text
agent 命令结束
  -> 收集 patch
  -> 关闭反向隧道
  -> DELETE agent sandbox
  -> 创建并运行 clean grader sandbox
  -> DELETE grader sandbox
  -> drain 并关闭 model proxy
```

异常或 rollout 取消时，task cancellation 会先向内传播；`finally` 删除 sandbox，
`AsyncExitStack` 再关闭 proxy。外层会等待 task 完成 unwind 后才进入 SGLang offload，避免
“Python task 已取消，但 sandbox、隧道和模型请求仍在后台运行”。

实际长跑日志能够看到 sandbox `DELETE` 请求返回 HTTP 204，说明这里执行的是平台侧立即
删除，不是只在本地丢掉对象、等待 TTL 自动回收。

这一组修改同时降低了 sandbox 峰值占用和异常后的资源泄漏风险，也减少了上一节中
“rollout 已结束但底层模型请求还在运行”的机会。

## 4. 共享盘异步保存可能留下损坏的 checkpoint shard（少数情况， 可选）

### 现象

保存 checkpoint 时出现过：

```text
CheckpointException ranks: dict_keys([])
[enforce fail at inline_container.cc:672] unexpected pos 704 vs 598
```

错误发生在 `torch.distributed.checkpoint` 的异步分片写入阶段。它不是长样本 OOM，也不是
actor/critic 训练失败。失败后目录中可能已经存在一部分 shard，但没有最终 `.metadata`。
直接在这个目录上重试，会混入上一次未完成的文件。

### 原因

原写入逻辑直接以最终文件名打开并写入。共享文件系统发生短写、中断或并发写压力时，最终
路径上会留下半个 PyTorch inline container；同时 BytesIO 在一次失败后游标已经前移，直接
重试也可能从错误位置继续。

### 修改

主要修改文件：

```text
Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py
Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py
Megatron-LM/megatron/training/checkpointing.py
```

处理方式是：

1. 每个 shard 先写到目标目录中的临时文件。
2. 重试前把 BytesIO 游标重置到开头。
3. 写完执行 `fsync`，成功后使用 `os.replace()` 原子替换最终文件。
4. 写入失败时删除临时文件，最多重试指定次数。
5. 再次保存同一个 iteration 前，只清理没有 `.metadata` 的不完整目录。
6. 已经存在 `.metadata` 的有效 checkpoint 不会自动删除。

另外增加了 `MCORE_DIST_CKPT_THREAD_COUNT`，让共享盘环境可以显式限制每个 rank 的 writer
数量。我们的运行配置使用一个 writer 和三次写入尝试，优先保证 checkpoint 可恢复性。

## 5. PPO 恢复点需要同时检查 actor、critic 和 dataset state（可选）

PPO 和只训练 actor 的 GRPO 不同。一次完整恢复至少涉及：

```text
actor checkpoint
critic checkpoint
rollout/global_dataset_state_dict_<iteration>.pt
```

原始恢复流程主要依赖各自 tracker。异步保存失败时，actor 和 critic 不一定同时更新；如果
只检查其中一边，下一次启动可能加载不同 iteration，或者模型 checkpoint 已存在但 dataset
cursor 缺失。

我们在 PPO 启动脚本中增加了恢复前检查：

- actor 和 critic 的 tracker 必须都是数字并且完全相同；
- 两侧对应 iteration 的 `.metadata` 必须存在且非空；
- actor 侧对应的 rollout dataset state 必须存在且非空；
- resume 时保存目录必须仍然指向同一条 actor/critic lineage。

训练结束后再做一次相同检查。这样可以把“Ray job 正常退出”和“形成可恢复的 PPO 状态”区分
开来。

恢复任务还需要使用 checkpoint 中保存的 optimizer scheduler 参数。否则仅仅因为本次想
少跑几个 step，命令行构造出的总迭代数就可能与 checkpoint 不同，并触发：

```text
OptimizerParamScheduler: class input value 12800 and checkpoint value 32000
for total number of iterations do not match
```

恢复入口因此会传入 `--use-checkpoint-opt-param-scheduler`。这不会改变已经训练过的学习率
进度，只是避免用临时任务长度覆盖 checkpoint 的 scheduler 定义。
场景。


