# Qwen3.5-4B SWE PPO 8卡优化与Debug记录

最后更新：2026-08-07 UTC

本文档是当前8卡dual SWE-Dev PPO实验的参数事实源，包含已经实施的优化、历史故障、恢复点、验证命令和回退条件。

## P0：8卡checkpoint可重载性（已关闭）

这是当前最高优先级的恢复阻断问题。训练计算本身可以连续运行，但旧代码保存的8卡checkpoint可能在任务重启后破坏Adam状态，使actor无法执行下一次更新。

### 故障现象与范围

失败日志为`resume-8gpu_20260807_101916.log`，关键顺序如下：

1. actor和critic都成功加载iteration 20。
2. step 21的64条rollout全部完成，reward和log-prob统计正常。
3. critic step 21完成。
4. actor进入`HybridDeviceOptimizer -> torch.optim.Adam -> _fused_adam`后失败：

```text
RuntimeError: a Tensor with 32 elements cannot be converted to Scalar
```

该错误发生在optimizer更新阶段，不涉及SGLang KV池、sandbox API、router、reward、长样本或checkpoint写盘。`SGLANG_MEM_FRACTION=0.70`在本次运行中已经正常完成SGLang释放和Megatron唤醒。

### 为什么0到19正常，19到29也曾正常

| 阶段 | actor/critic卡数 | TP | 每侧DP | checkpoint来源 | 结果 |
| --- | ---: | ---: | ---: | --- | --- |
| 初始4卡训练 | 2 / 2 | 2 | 1 | HF转换后的release | 连续训练并形成有效iteration 19 |
| 第一次8卡恢复 | 4 / 4 | 2 | 2 | 4卡DP=1的iteration 19 | 同一进程成功完成step 20到30 |
| 8卡step 20验证 | 4 / 4 | 2 | 2 | iteration 19 | step 20训练和保存成功，之后SGLang因误传0.90 OOM |
| 8卡step 21验证 | 4 / 4 | 2 | 2 | 8卡DP=2保存的iteration 20 | 重载后actor首次更新触发32元素step错误 |
| 修复后step 21验证 | 4 / 4 | 2 | 2 | 含旧错误step的iteration 20 | actor/critic更新、保存和SGLang恢复全部成功，形成iteration 21 |
| 二次重载step 22 | 4 / 4 | 2 | 2 | 修复后生成的iteration 21 | actor/critic更新成功，原32元素错误未再出现 |
| 连续step 23 | 4 / 4 | 2 | 2 | 同一长跑进程 | actor/critic更新成功，无OOM或系统错误 |
| 5-step周期保存step 24 | 4 / 4 | 2 | 2 | 同一长跑进程 | actor/critic成对保存，tracker/metadata/dataset state完整，SGLang全部恢复 |

具体原因：

- iteration 19由4卡布局保存。actor和critic各2卡，`TP=2`后每侧只有`DP=1`，没有本次DP=2对齐产生的32元素padding。
- 从19切换到8卡时，加载的是干净的DP=1 checkpoint。内存中的Adam `step`仍是0维标量，所以训练可以继续。
- `resume-8gpu_20260807_061859.log`已经证明8卡同一进程完成了step 20到30。step号本身没有问题。
- 保存checkpoint只把当前状态写到磁盘，不会立刻把磁盘内容重新加载回正在运行的optimizer。即使旧保存逻辑写出了错误的重复`step`，当前进程中的标量仍然正确，因此可以继续跑后续step。
- 问题只会在任务退出后，从一个由8卡`DP=2`保存的checkpoint重新加载时暴露。

因此故障边界不是“step 19以后不能训练”，而是：**旧代码生成的8卡DP=2 checkpoint不能保证被下一次任务正确重载。**

### 正常Adam状态应该是什么

PyTorch Adam为每个参数维护的主要状态可抽象为：

```python
optimizer.state[param] = {
    "step": scalar_tensor,       # 0维标量，例如 tensor(20.)
    "exp_avg": parameter_tensor,
    "exp_avg_sq": parameter_tensor,
    "master_param": parameter_tensor,
}
```

在Megatron distributed optimizer checkpoint中，两类数据必须分开：

- `step`是统一的optimizer迭代计数，应通过optimizer `param_groups`保存和恢复。
- `exp_avg`、`exp_avg_sq`和`master_param`与参数同形状，才应该进入`dp_reshardable` per-parameter bucket state并按DP切分。

旧代码同时把`step`放进了这两个位置。param_groups中的副本正确，但per-parameter state中的重复副本会在DP padding阶段被错误扩展。

### 32元素tensor如何产生

保存路径位于：

```text
DistributedOptimizer.get_parameter_state_dp_reshardable()
  -> DistributedOptimizer.sharded_param_state_dp_reshardable()
  -> torch distributed checkpoint writer
```

旧的`get_parameter_state_dp_reshardable()`调用`_get_main_param_and_optimizer_states()`后保留了所有tensor，包括0维`step`。随后`sharded_param_state_dp_reshardable()`为了覆盖DP bucket空隙，会为每个tensor创建padding：

```python
pad_tensors = {
    key: torch.empty(padding_length, dtype=value.dtype, device=value.device)
    for key, value in parameter_state.items()
    if isinstance(value, torch.Tensor)
}
```

当本次bucket空隙为32时：

- 正常的`exp_avg`等状态得到32元素padding，这是预期行为。
- 标量`step`也被错误地执行`torch.empty(32)`，变成未初始化的32元素tensor。
- 后续代码对`step`只是`continue`，没有把它从state dict删除，因此错误tensor仍进入checkpoint common/merge结构。

加载路径为：

```text
DistributedOptimizer.load_state_dict()
  -> 从param_groups恢复正确标量step
DistributedOptimizer.load_parameter_state_from_dp_reshardable()
  -> _set_main_param_and_optimizer_states()
  -> 错误的per-parameter step再次覆盖正确标量
HybridDeviceOptimizer._sync_hdo_state_to_sub_optimizers()
  -> 错误step进入CPU Adam子优化器
torch.optim.Adam._fused_adam()
  -> 需要scalar，收到32元素tensor并退出
```

这也解释了为什么checkpoint加载阶段显示`successfully loaded`，真正错误却延迟到actor第一次`optimizer.step()`才出现：加载器只检查tensor能否装入结构，没有检查Adam `step.numel() == 1`。

日志中critic先完成而actor先报错，只表示actor最早走到了包含错误state的CPU-offload参数；不能据此认为critic checkpoint一定不受影响。因此保存端和加载端修复同时对actor、critic生效。

### 代码如何修复

修改文件：

```text
Megatron-LM/megatron/core/optimizer/distrib_optimizer.py
```

#### 1. 保存端：禁止step进入per-parameter state

在`get_parameter_state_dp_reshardable()`中，取得parameter state后立即移除`step`：

```python
tensors = self._get_main_param_and_optimizer_states(model_param)
tensors.pop("step", None)
```

修复效果：

- padding生成器永远看不到`step`，不会再创建parameter-shaped step。
- `step`仍由现有`DistributedOptimizer.state_dict()`写入param_groups，没有丢失optimizer迭代数。
- `exp_avg`、`exp_avg_sq`和`master_param`的保存路径、dtype、shape和分片方式完全不变。

#### 2. 加载端：兼容旧的iteration 20

在`load_parameter_state_from_dp_reshardable()`调用`_set_main_param_and_optimizer_states()`之前过滤`step`：

```python
src_tensors = {key: value for key, value in src_tensors.items() if key != "step"}
self._set_main_param_and_optimizer_states(model_param, src_tensors)
```

修复效果：

- 新checkpoint没有重复`step`，正常加载。
- 旧iteration 20即使包含32元素`step`，也会在覆盖optimizer state前被丢弃。
- `load_state_dict()`此前已从param_groups恢复正确标量，所以不是重置optimizer step，也不是从零创建新optimizer。
- 模型权重、Adam moments、master parameter、学习率scheduler和dataset cursor继续从iteration 20严格恢复。
- 无需删除iteration 20，也无需回退到19重新生成rollout。

#### 3. 回归测试

新增测试位于：

```text
Megatron-LM/tests/unit_tests/dist_checkpointing/test_optimizer.py
```

两个测试分别验证：

1. 保存端生成的`dp_reshardable` parameter state不再包含`step`。
2. 加载旧checkpoint时，即使输入显式包含`torch.zeros(32)`形式的错误`step`，传给`_set_main_param_and_optimizer_states()`的状态中也不会包含它。

测试已在精确训练镜像`avatrain-miles:...cu129`中执行：

```text
2 passed
```

同时已通过Python静态编译、Bash语法检查和`git diff --check`。

### 为什么现有iteration 20可以继续使用

iteration 20不是整体损坏：

- actor和critic的`.metadata`完整，tracker都指向20。
- rollout dataset state 20完整。
- 模型权重以及Adam `exp_avg`、`exp_avg_sq`、master weights已经成功写入。
- optimizer param_groups中保存了正确的标量step。
- 唯一需要忽略的是per-parameter state中多余且形状错误的重复`step`。

因此加载端过滤属于定向兼容修复，不是跳过全部optimizer state，也不会把PPO改成只恢复模型权重。

### P0关闭条件与结果

不能仅凭单测宣布问题完全关闭，必须完成两次真实8卡验证：

1. 已完成：从iteration 20运行step 21并成对保存iteration 21，证明旧checkpoint兼容加载成功。
2. 已完成：新任务从修复后生成的iteration 21加载，actor/critic step 22和step 23均成功，证明新checkpoint可再次加载。
3. 已完成：`SAVE_INTERVAL=5`的首个全局周期点iteration 24成对落盘，并在保存后成功恢复8个SGLang engine。

两次运行都必须满足：

- actor和critic训练日志同时出现对应step。
- 不出现`Tensor with 32 elements cannot be converted to Scalar`。
- 单步验证时actor/critic tracker相同，两侧`.metadata`和actor dataset state存在且非空；长跑时按5步周期检查。
- SGLang恢复阶段不出现`cuMemCreate`；显存比例保持`0.70`。

三项条件均已满足，因此该P0于2026-08-07 UTC关闭。对应长跑日志为`resume-8gpu_20260807_110439.log`。

## 当前恢复事实

- actor和critic的`latest_checkpointed_iteration.txt`均为`24`。
- 两边的`iter_0000024/.metadata`均存在且非空。
- actor侧`rollout/global_dataset_state_dict_24.pt`存在且非空。
- 日志曾训练完成step 30，但没有形成成对checkpoint，因此不能从30恢复。
- 2026-08-07 09:58运行已完成step 20的rollout、训练和成对checkpoint保存，随后在恢复SGLang KV池时OOM。
- 2026-08-07 10:19运行从step 20恢复，step 21的64条rollout和critic更新完成；actor首次Adam更新发现checkpoint中的非标量`step`后退出，因此没有保存step 21，恢复点仍为20。
- 2026-08-07 10:45修复后再次从step 20恢复，actor/critic step 21、成对checkpoint保存、权重更新和SGLang恢复全部成功。
- 2026-08-07 11:04长跑从iteration 21恢复，step 22、23、24的actor/critic更新全部成功。
- iteration 24于11:33 UTC完成actor/critic成对保存，11:34 UTC完成权重更新和8个SGLang engine恢复。
- step 25的64条rollout、actor/critic更新、权重更新和8个SGLang engine恢复也已全部成功；当前已进入step 26 rollout。
- 若任务现在中断，正确恢复点是iteration 24，下一次执行step 25。
- `iter_0000029.failed_20260807_061356`是不完整actor保存，不在tracker引用范围内。
- 单侧checkpoint约67 GiB，actor和critic成对保存约134 GiB。

## 当前有效参数

| 类别 | 参数 | 有效值 | 说明 |
| --- | --- | ---: | --- |
| GPU布局 | actor / critic | 4 / 4 | 共8张H100 |
| 训练并行 | TP | 2 | 保持checkpoint的TP结构；actor和critic各DP=2 |
| Rollout | engine | 8个TP=1 | colocate阶段每张卡一个SGLang engine |
| Batch | rollout / global | 64 / 64 | 每64条trajectory更新一次PPO |
| 数据 | dual SWE-Dev | 1000 | primary 500 + secondary 500 |
| Sandbox | primary / secondary | 32 / 32 | 一批最多同时推进64个episode |
| SGLang | server concurrency | 32 | 提高rollout请求供给 |
| SGLang | memory fraction | 0.70 | 为Megatron与SGLang恢复KV池的共置峰值留出空间 |
| 路由 | policy | cache_aware | 利用prefix cache并分散请求 |
| 路由 | cache / abs / rel阈值 | 0.30 / 8 / 1.25 | 保留前缀复用，但更早纠正单卡排队 |
| 长度 | trajectory响应总长 | 16384 | 包含模型输出和后续工具观察tail |
| 长度 | 单次模型生成 | 8192 | 不是训练侧2048分块 |
| 训练显存 | max tokens/GPU | 2048 | dynamic batch训练分块 |
| 训练显存 | log-prob chunk | 16 | 限制全词表logits临时张量 |
| 训练显存 | margin | 512 MiB | backward前显存保护 |
| Optimizer | resume offload | 0.1 | 必须与当前checkpoint lineage一致 |
| Checkpoint | 长跑保存间隔 | 5 | 每5步成对保存，在写盘成本和故障回退距离之间折中 |
| Checkpoint | writer / retry | 1 / 3 | 每rank单writer，失败最多3次 |
| Ray | CPU | 16 | 8卡默认每卡2核 |

## 参数一致性审计

参数经过三层核对：

1. `scripts/train/run_ppo.sh`选择8卡布局、resume checkpoint和实验默认值。
2. `miles/examples/agentic_swe/run_qwen35_4b_4gpu_ppo.sh`把值转换为Miles/Megatron CLI参数。
3. Ray `runtime_env`传递Agent、sandbox和checkpoint writer环境变量。

已消除的漂移：

- dual顶层和PPO后端的`SGLANG_MEM_FRACTION`统一为`0.70`。
- 8卡启动前拒绝`SGLANG_MEM_FRACTION>0.70`，避免rollout和训练完成后才在KV恢复阶段崩溃。
- dual顶层和PPO后端的`SGLANG_SERVER_CONCURRENCY`统一为`32`。
- PPO后端的Ray CPU默认值改为`NUM_GPUS * 2`，8卡得到16，4卡得到8。
- `SAVE_INTERVAL=5`、response 16K、训练分块2048、log-prob chunk 16在两层一致；smoke测试单独强制保存间隔1。
- checkpoint writer的`1/3`通过Ray runtime env传给Megatron actor。
- sandbox `32/32`通过Ray runtime env传给rollout worker。
- 启动前新增校验：dual sandbox并发总数不能超过本批trajectory总数。

以下差异是有意设计，不是参数错误：

- fresh训练optimizer offload默认`0.4`；当前resume lineage强制`0.1`以匹配checkpoint结构。
- dual使用`0.70/32/32+32`；secondary-only使用`0.70/12/0+64`，二者API和数据布局不同。
- 单次Agent生成8192、trajectory总长16384、训练分块2048分别控制不同阶段。
- 已完成的单步验证使用`SAVE_INTERVAL=1`；当前长跑显式使用`SAVE_INTERVAL=5`。

## 已实施的利用率优化

### 1. Rollout供给

- 每批仍为64条，不改变PPO batch语义。
- primary和secondary分别使用独立semaphore，默认`32/32`。
- SGLang服务并发提高到32，8张卡运行8个TP=1 engine。
- 使用`cache_aware`路由，避免所有请求长期集中在一张卡。
- 旧默认平衡阈值为`abs=64, rel=1.5`，对每批64条的负载倾斜反应过迟；现收紧为`abs=8, rel=1.25`。
- 当最忙与最空worker差距超过8且相对负载超过1.25倍时，router优先再平衡；否则保留cache-aware的长前缀复用收益。
- sandbox创建具有指数退避和最多120次重试，短时API限流不会立即终止整批。

历史日志确认router并非只打印配置：

- 训练CLI包含`--router-policy cache_aware`，最终参数表为`router_policy=cache_aware`。
- router启动记录明确显示`RouterArgs(... policy='cache_aware' ...)`。
- bulk rollout阶段多次出现`Decode batch ... [repeated 8/9x across cluster]`和`POST /generate ... [repeated 7/9x across cluster]`，说明多个engine同时处理请求。
- Ray默认合并相似worker日志，因此不能按日志表面PID行数统计每卡请求量；大量行显示在一个PID下不表示请求只去了那张卡。
- rollout尾部若只剩一个长trajectory，单个TP=1请求只能运行在一张卡上，任何router都无法把同一条序列拆给8个独立engine。

step 24给出了一个可复现的长尾例子：63条已经完成后，`pre-commit__pre-commit-1415`的grader等待到`SWE_GRADER_TIMEOUT_SEC=600`才以reward 0结束，使整个rollout耗时631.2秒。这是评测长尾，不是GPU、router或checkpoint卡死；期间8个SGLang health持续返回200。超时后sandbox被DELETE 204回收，该step仍正常进入训练和保存。

iteration 25启动时sandbox API曾短暂返回429/500 capacity exhausted。已有退避重试生效，同一轮随后返回201并成功创建；这种短时容量压力不需要手工停任务或删除checkpoint。

注意：Agent运行shell、测试和grader时不调用GPU。该阶段GPU利用率下降是Agentic RL工作负载特性，不是少启动了GPU。日志中训练阶段actor吞吐达到约18K到28K token/s，但step 29/30的等待占比分别约87%和95%，说明主要瓶颈是sandbox长尾而非模型计算。

### 2. 长样本处理

- 完整trajectory响应预算硬限制为16K。
- 每轮根据剩余总预算动态计算`max_new_tokens`。
- 工具输出导致最终ledger超过16K时，只截断超出的尾部。
- 截断同时裁剪`tokens`、`loss_mask`和`rollout_log_probs`，保持严格对齐。
- 被截断样本标记为`Sample.Status.TRUNCATED`；未超长样本不受影响。

### 3. SGLang残留请求

- episode关闭时取消仍在等待的proxy future。
- 模型offload前调用`/abort_request`并设置`abort_all=true`。
- `/flush_cache`按真实单调时钟等待60秒。
- HTTP 400时每秒重试一次，不再同一秒空转60次后产生假超时。

### 4. 训练显存

- dynamic batch按每GPU 2048 token切分。
- log-prob使用16-token chunk，降低Qwen大词表临时张量峰值。
- full recompute、sequence parallel和512 MiB训练margin保持启用。
- actor和critic各使用4卡、TP=2，不改变checkpoint TP结构。
- resume optimizer CPU offload固定0.1，避免optimizer state结构不匹配。

### 5. Checkpoint可靠性

- 共享virtiofs上每rank仅使用一个分片writer。
- 分片先写同目录临时文件，`fsync`成功后使用`os.replace`原子落盘。
- 写失败会删除临时文件、回卷BytesIO并最多重试3次。
- 重试同一iteration前，只删除没有`.metadata`的不完整目录。
- resume加入`--use-checkpoint-opt-param-scheduler`，解决`12800 vs 32000` scheduler断言。
- actor、critic和dataset state必须使用相同iteration才算有效恢复点。

## 历史故障与结论

### `cuMemCreate CUDA_ERROR_OUT_OF_MEMORY`

历史`mem_fraction=0.85`在`torch_memory_saver.cpp::resume`失败。2026-08-07的step 20验证误传了`0.90`，SGLang实际预留约73438 MB；actor/critic保存成功后，恢复静态KV池再次触发`cuMemCreate CUDA_ERROR_OUT_OF_MEMORY`。16K截断只能限制长请求，不能降低SGLang静态KV池预留。当前固定使用`0.70`，sandbox和server并发仍保持`32/32`和`32`。

### `Timeout while flushing cache`

原因是客户端超时后SGLang仍保留孤儿请求，加上旧重试循环没有sleep。已通过abort-all和真实60秒等待修复。

### `CheckpointException`和`unexpected pos`

原因发生在共享盘多进程写PyTorch inline container阶段。已通过单writer、临时文件、原子替换、重试和不完整目录清理加固。

### Optimizer scheduler mismatch

checkpoint保存的总迭代数为32000，而短恢复任务构造出12800。resume现在使用checkpoint scheduler参数，不再因`NUM_ROLLOUT`缩短而拒绝加载。

### `Tensor with 32 elements cannot be converted to Scalar`

这是Megatron `HybridDeviceOptimizer`与`dp_reshardable` checkpoint组合的optimizer-state bug，不是SGLang、sandbox、样本长度或reward问题。

- Adam的`step`本应是0维标量，并通过optimizer `param_groups`保存。
- 旧实现仍把`step`留在per-parameter bucket state中；构造bucket padding时又按padding长度创建了`step`，本次长度恰好为32。
- step 20恢复时，`load_parameter_state_from_dp_reshardable()`把这个32元素向量覆盖到actor Adam state。
- critic完成更新后，actor的PyTorch fused Adam尝试读取标量`step`，因此在`_fused_adam`退出。

修复同时覆盖新旧checkpoint：

- 保存端在构造`dp_reshardable` parameter state时移除`step`，以后不会再生成parameter-shaped step。
- 加载端无条件忽略旧parameter state中的`step`，保留已从optimizer param_groups恢复的正确标量。
- `exp_avg`、`exp_avg_sq`、master parameter、optimizer scheduler和模型权重均照常严格加载。
- step 20无需删除或回退；兼容修复已生成iteration 21并通过step 22二次重载验证。当前最新有效恢复点为iteration 24。

## 已完成的Step 21兼容验证命令

在8xH100作业中执行：

```bash
cd /inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/swe-rl/AvaTrain

SGLANG_MEM_FRACTION=0.70 \
SGLANG_SERVER_CONCURRENCY=32 \
ROUTER_BALANCE_ABS_THRESHOLD=8 \
ROUTER_BALANCE_REL_THRESHOLD=1.25 \
SANDBOX_CONCURRENCY_PRIMARY=32 \
SANDBOX_CONCURRENCY_SECONDARY=32 \
RAY_NUM_CPUS=16 \
NUM_ROLLOUT=22 \
SAVE_INTERVAL=1 \
MCORE_DIST_CKPT_THREAD_COUNT=1 \
MCORE_DIST_CKPT_WRITE_ATTEMPTS=3 \
bash scripts/train/run_ppo.sh resume-8gpu
```

`NUM_ROLLOUT=22`表示从iteration 20恢复后只运行step 21。

## Step 21成功条件

```bash
ACTOR=/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/swe-rl/checkpoints/swe_dev_dual_qwen35_4b_ppo_actor_20260805_133102
CRITIC=/inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/swe-rl/checkpoints/swe_dev_dual_qwen35_4b_ppo_critic_20260805_133102

cat "$ACTOR/latest_checkpointed_iteration.txt"
cat "$CRITIC/latest_checkpointed_iteration.txt"
test -s "$ACTOR/iter_0000021/.metadata"
test -s "$CRITIC/iter_0000021/.metadata"
test -s "$ACTOR/rollout/global_dataset_state_dict_21.pt"
```

两个tracker都必须输出`21`，三个`test`都必须返回0。

## 已验证的Step 22二次重载与长跑命令

```bash
cd /inspire/hdd/project/exploration-topic/public/sywang/fzk/slime/examples/swe-rl/AvaTrain

SGLANG_MEM_FRACTION=0.70 \
SGLANG_SERVER_CONCURRENCY=32 \
ROUTER_BALANCE_ABS_THRESHOLD=8 \
ROUTER_BALANCE_REL_THRESHOLD=1.25 \
SANDBOX_CONCURRENCY_PRIMARY=32 \
SANDBOX_CONCURRENCY_SECONDARY=32 \
RAY_NUM_CPUS=16 \
NUM_ROLLOUT=500 \
SAVE_INTERVAL=5 \
MCORE_DIST_CKPT_THREAD_COUNT=1 \
MCORE_DIST_CKPT_WRITE_ATTEMPTS=3 \
bash scripts/train/run_ppo.sh resume-8gpu
```

## 监控与回退

```bash
LOG=$(ls -1t /inspire/qb-ilm/project/exploration-topic/public/sywang/fzk/logs/swe-rl/swe_dev_dual-ppo-gpu/resume-8gpu_*.log | head -1)
tail -F "$LOG"
```

健康日志应持续出现递增的`rollout N`、`critic-step N`和`step N`。

出现以下任一错误时停止并检查：

```text
CUDA_ERROR_OUT_OF_MEMORY
cuMemCreate
CheckpointException
Timeout while flushing cache
```

若仍出现`cuMemCreate`或SGLang resume OOM，不要提高显存比例；保留日志并检查是否有外部GPU进程或offload失效。sandbox并发控制请求供给，不等于静态KV显存比例，不需要因为本次OOM而降低。
