# MGV 方法文档

这份文档只描述当前仍在使用的 StarVLA MGV 与 RLinf 在线 RL 实现，不再记录已经删除的兼容路径。

## 1. 离线：StarVLA 中的 MGV

### 1.1 总体结构

当前训练显式拆成两个 forward：

- `forward_mode="action"`：每个训练 step 都执行，负责动作学习。
- `forward_mode="mgv"`：只在满足 `mgv_forward_interval` 时执行，负责可达性价值学习。

入口在 [QwenOFT.py](../starVLA/model/framework/VLM4A/QwenOFT.py)。`action` 和 `mgv` 不再混在一次 forward 里，各自单独构造输入、单独算 loss。

### 1.2 MGV 额外 special tokens

定义在 [mgv_progress.py](../starVLA/model/modules/mgv_progress.py)：

- `<state_start>`
- `<state_end>`
- `<goal_img_start>`
- `<goal_img_end>`
- `<goal_lang_start>`
- `<goal_lang_end>`
- `<value>`

这些 token 通过 `additional_special_tokens` 注册到 tokenizer，并在模型 embedding 表里新增可训练行。

### 1.3 Action 分支

Action 分支只会二选一使用 goal。

1. image-goal action：

```text
[state images] +
"<state_start> ... <state_end>" +
[future goal images] +
"<goal_img_start> <goal_img_end>" +
"Please predict the next N robot actions: <action>🔍...🔍<action>."
```

2. lang-goal action：

```text
[state images] +
"<state_start> ... <state_end>" +
"<goal_lang_start> {language goal} <goal_lang_end>" +
"Please predict the next N robot actions: <action>🔍...🔍<action>."
```

训练时通过 `action_goal_lang_prob` 控制两种 action goal 的采样比例：

- `1.0`：action 永远只看 language goal。
- `0.0`：action 永远只看 future image goal。
- 中间值：按概率采样。

如果 `batch_consistent_action_goal=true`，整批样本共享一次采样；否则逐样本采样。

`terminal_goal_obs` 不进入 action 分支。`predict_action()` 推理时固定走 lang-goal 路径，也就是部署时默认输入始终是 `state + lang goal + action prompt`。

### 1.4 MGV 分支

MGV forward 不预测 action，而是单独构造 value query。value instruction 固定为：

```text
Predict the discounted reachability value from the current observation to the goal. <value>
```

当前会分别构造三类 goal：

- future image goal
- terminal image goal
- language goal

对应三种 value 输入：

```text
[state] + [future image goal] + [value prompt]
[state] + [terminal image goal] + [value prompt]
[state] + [language goal] + [value prompt]
```

三类 goal 是分别前向编码的，不会互相出现在同一个 value 序列里。

### 1.5 Value head

MGV head 的结构在 [mgv_progress.py](../starVLA/model/modules/mgv_progress.py)。

先从 backbone hidden states 中取出：

- `hidden(<state_end>)`
- `hidden(<goal_img_end>)` 或 `hidden(<goal_lang_end>)`
- `hidden(<value>)`

再分别通过：

- `state_proj`
- `img_goal_proj`
- `lang_goal_proj`
- `value_proj`

最后把三路特征拼接，输入：

- `value_projector`

输出一个 scalar logit。最终用 `sigmoid(logit)` 得到 `[0, 1]` 范围的 reachability prediction。

### 1.6 离线监督目标

#### 1.6.1 Future image / terminal image reachability

当前监督目标是折扣可达性：

```text
target = gamma ^ K
```

其中：

- future image goal: `K = floor((h - k) / stride)`
- terminal goal: `K = floor((T - k) / stride)`

这里的 `k / h / T / mgv_temporal_stride` 必须由数据集 metadata 显式提供；当前实现不再允许 silent fallback。

对应 loss 是：

```text
BCEWithLogits(logit, target)
```

#### 1.6.2 Lang-term distill

如果同时有 `g_lang` 和 `g_term`，则额外计算：

- `logit_lang`
- `logit_term`

默认主方向是 `lang <- term`：

```text
BCEWithLogits(
    logit_lang,
    sigmoid(stop_gradient(logit_term))
)
```

也就是让 language-goal reachability 去拟合 terminal-goal reachability 的 teacher probability，而不是直接做 logit MSE。

如果打开反向蒸馏，也可以额外做 `term <- lang`。是否启用由：

- `enable_distill_term_to_lang`
- `enable_distill_lang_to_term`

控制，总权重由 `eta_distill` 控制。

#### 1.6.3 Value norm regularization

当前实现仍保留一项轻量 norm penalty：

```text
1e-4 * mean(||z||^2)
```

它作用在参与 MGV 的投影向量上，用来抑制 value 表征范数无约束增大。

### 1.7 离线总 loss

局部 MGV loss 为：

```text
loss_mgv =
    loss_img_reach
  + loss_term_reach
  + eta_distill * loss_distill
  + loss_value_norm
```

trainer 侧再乘 `lambda_mgv`，并与 action loss 一起优化。

### 1.8 离线关键指标

常用指标包括：

- `L_act`：action L1 loss
- `L_mgv`：总 MGV loss
- `L_img_reach`：future image reachability BCE
- `L_term_reach`：terminal reachability BCE
- `L_distill`：lang/term distill loss
- `L_value_norm`：value 表征 norm 正则
- `img_reach_pred_mean` / `img_reach_target_mean`
- `term_reach_pred_mean` / `term_reach_target_mean`
- `distill_lang_reach_mean`
- `distill_term_reach_mean`
- `distill_reach_lang_term_mae`
- `raw_z_*_norm_mean`

这些指标的核心作用是分别看 action 拟合、image/terminal reachability 拟合、lang-term 对齐，以及 value 表征的数值范围是否稳定。

## 2. 在线：RLinf 中的 off-policy MGV

### 2.1 总体结构

在线部分位于 RLinf 仓库中，当前入口是：

- [fsdp_mgv_trl_policy_worker.py](../../RLinf_offpolicy_launch_20260608/rlinf/workers/actor/fsdp_mgv_trl_policy_worker.py)
- [starvla_action_model.py](../../RLinf_offpolicy_launch_20260608/rlinf/models/embodiment/starvla/starvla_action_model.py)

当前实现已经不是早期“mode A / mode B 分开训练”的版本，而是统一成一个 replay-driven 的 off-policy trainer。每次更新都从 replay buffer 采样一批 merged batch，同时驱动：

- image-goal transitive value 学习
- success 条件下的 `g_lang` distill 与 `g_lang` transitive 学习
- 可选的 `Q(s, a, g)` 学习
- 可选的 actor 更新

在线部分的两个主开关是：

- `actor.model.actor_policy_family`: `gaussian` / `deterministic`
- `algorithm.actor_update_mode`: `awac` / `ddpg_bc`

另外还有一个 AWAC 专用开关：

- `algorithm.actor_weight_target_mode`: `advantage` / `next_value`

以及一个 Q-learning 开关：

- `algorithm.enable_q_learning`

其中：

- `ddpg_bc` 要求 `enable_q_learning=True`
- `awac` 可以只用 `V`，也可以同时训练 `Q` 但不使用 `Q` 更新 actor

### 2.2 在线可训练参数范围

当前 worker 会强制把 `rl_trainable_scope` 设成 `actor_mgv_heads`，因此不会训练 StarVLA 的大 VLM 主干。

当前允许更新的参数是：

- `starvla_model.action_model.*`
- `starvla_model.action_expert.*`
- `starvla_model.mgv_module.state_proj.*`
- `starvla_model.mgv_module.img_goal_proj.*`
- `starvla_model.mgv_module.lang_goal_proj.*`
- `starvla_model.mgv_module.value_proj.*`
- `starvla_model.mgv_module.value_projector.*`
- `mgv_q_delta_encoder.*`（如果启用）
- `actor_logstd`（仅高斯策略）

因此，在线阶段的学习对象本质上是：

- action head
- MGV head
- 可选 Q delta encoder
- 可选高斯方差参数

### 2.3 Target network

当前实现维护两套轻量 target：

1. `target_value_head`

- 只复制 MGV head 的投影与 `value_projector`
- 不复制整套 VLM 主干
- 每次 optimizer step 后用 `target_tau` 做 soft update

2. `target_actor_action_model`

- 只复制 `action_model`
- 只在 `ddpg_bc` 路径中作为 BC regularizer 的 teacher 使用
- 同样每次 optimizer step 后做 soft update

当前没有单独的 target-Q 网络，也没有 target delta encoder。`Q(s, a, g)` 的监督信号来自 target value head 在 `next_obs=s'` 上的 bootstrap：

```text
Q_target(s, a, g)
    = gamma_chunk * stop_gradient( V_target(s', g) )
```

如果 `d(s', g)` 足够小，还会先把右侧的 `V_target(s', g)` 替换成 exact reachable target，再乘 `gamma_chunk`：

```text
V_rhs(s', g) = gamma_chunk ^ d(s', g)
Q_target(s, a, g) = gamma_chunk * V_rhs(s', g)
```

其中：

- `g_img` 可以直接用 `d(s', g_img)` 做 exact override
- `g_lang` 只有 success sample 才会做 exact override，因为只有这时 `g_lang` 与采样到的成功 goal 对齐

### 2.4 在线 value 表示与离线保持一致

RLinf 没有重写 value encoder，而是直接复用 StarVLA checkpoint 内的 `mgv_module`：

- 仍然使用同一套 special tokens
- 仍然使用同一句 value prompt
- 仍然抽取 `hidden(<state_end>)`、`hidden(<goal_x_end>)`、`hidden(<value>)`
- 仍然走 `state_proj / goal_proj / value_proj / value_projector`

因此，在线 `V(s, g)` 与离线 MGV 在输入形式和 head 结构上是一致的。

需要注意的是：

- 底层 `starvla_action_model.py` 支持 `future_image / terminal_image / lang`
- 但当前在线 worker 实际使用的 goal 只有 `future_image` 和 `lang`
- `terminal_image` 当前不参与 RLinf 这套 off-policy 更新

### 2.5 Replay、rollout 与 merged sampling

在线训练循环是：

1. rollout worker 用当前 policy 与环境交互
2. 将轨迹写入 replay buffer
3. actor worker 从 replay buffer 反复采样并更新，`utd_ratio` 可以大于 1

rollout 侧当前会缓存：

- `curr_obs`
- `next_obs`
- `forward_inputs`
- `success_once`

其中对 action 相关缓存要区分两种：

- `forward_inputs["action"]`：反归一化后的 env action
- `forward_inputs["action_for_logprob"]`：模型动作空间中的 executed action，用于高斯 logprob 或 deterministic supervised target

当前 merged batch 采样不再区分旧文档里的 mode A / mode B，而是统一采样：

```text
(s_i, s_j, s_k, s'_i, g_lang, is_success_sample)
```

采样规则是：

- 先按 `success_sample_prob` 决定是否采 success 样本
- 若采 success 样本，则 `k` 固定为该轨迹第一次成功后的 coarse state
- 否则 `k` 从任意未来 coarse state 中随机采样
- `i` 从 `[0, k)` 采样
- `j` 从 `[i + 1, k]` 采样

因此当前允许：

- `s_j == s_k`
- `d_jk == 0`

但不允许：

- `s_i == s_k`

同时会返回：

- `d_ij`
- `d_jk`
- `d_ik`
- `is_success_sample`

并且经常会派生使用：

- `d_spk = max(d_ik - 1, 0)`，表示 `s'_i -> g` 的 chunk-level 距离

其中 success 样本会额外启用 `g_lang` 的 distill 和 transitive loss。

### 2.6 Rollout 时的策略采样

当前 rollout 仍然走 StarVLA 原有 action 推理路径，但是否真的有随机性取决于 actor family：

1. `gaussian`

- 先由 StarVLA action head 产生 mean action
- RLinf 额外维护一个可学习的 `actor_logstd`
- rollout 时若 `sampling_params.do_sample=True` 且 `mode="train"`，则从：

```text
Normal(mean, std)
```

采样 executed action

2. `deterministic`

- rollout 直接使用 mean action
- 即使 `do_sample=True` 也不会引入额外随机性，因为 deterministic 路径没有高斯分布对象

当前 AWAC 配置也已经统一设成 `sampling_params.do_sample=True`。因此：

- 高斯 AWAC 现在会按高斯策略实际采样 rollout
- deterministic AWAC 仍然是确定性 rollout

### 2.7 在线 value 学习

#### 2.7.1 Future-image transitive loss

先预测：

```text
logit_ik = V(s_i, s_k)
```

再构造两段 bridge target：

```text
target_ij = exact_or_target( V(s_i, s_j) )
target_jk = exact_or_target( V(s_j, s_k) )
bridge_target = stop_gradient(target_ij * target_jk)
```

其中 `exact_or_target` 的规则是：

- 若距离 `d <= short_horizon_M_chunk`，直接用 exact target
- 否则用 `target_value_head` 预测的概率

exact target 具体为：

```text
gamma_primitive ^ (d * chunk_len)
```

也等价于：

```text
gamma_chunk ^ d
```

对 `V(s_i, s_k)` 的训练规则是：

- 短程：`BCEWithLogits(logit_ik, exact_target_ik)`
- 长程：`expectile_BCE(logit_ik, bridge_target)`

这里的 `expectile_BCE` 是当前实现中的 TRL-style implicit-max surrogate。

此外还会做 distance reweighting：

1. 先由 `sigmoid(logit_ik)` 反推出估计距离 `d_hat`
2. 再按 `1 / (1 + d_hat)^lambda` 给 transitive loss 加权

最终得到：

```text
loss_transitive
```

#### 2.7.2 Success 样本上的 `g_lang` distill

只对 `is_success_sample=True` 的样本，额外训练：

```text
V(s_i, g_lang)
```

去拟合：

```text
V(s_i, s_k)
```

目标构造规则是：

- 若 `d_ik <= short_horizon_M_chunk`，直接用 exact `gamma_chunk ^ d_ik`
- 否则用 `target_value_head` 对 `future_image` goal 的预测值

对应 loss 是：

```text
loss_lang_distill =
    BCEWithLogits( logit_lang_s, target_lang )
```

这部分可以理解为在线版的：

```text
g_lang <- success image
```

#### 2.7.3 Success 样本上的 `g_lang` transitive loss

当前实现里，`g_lang` transitive 不再只是 metric，而是实际进入 loss。

做法是先预测：

```text
V(s_i, g_lang)
```

再构造：

```text
bootstrap_lang_target =
    stop_gradient( target_ij * V_target(s_j, g_lang) )
```

其中：

- 短程 success 样本仍用 exact `gamma_chunk ^ d_ik`
- 长程 success 样本才用上面的 bridge target

训练方式与 image transitive 类似：

- 短程：普通 BCE
- 长程：expectile-weighted BCE

最终得到：

```text
loss_lang_transitive
```

#### 2.7.4 当前在线 value 总损失

value 相关损失项是：

```text
lang_distill_coef * loss_lang_distill
+ transitive_coef * (loss_transitive + loss_lang_transitive)
```

### 2.8 可选 Q 学习

当 `enable_q_learning=True` 时，额外学习：

```text
Q(s, a, g)
```

当前 `Q` 不是一套大网络，而是：

1. 先取出 `hidden(<state_end>) = h_state`
2. 把 action chunk 展平
3. 用一个小型 MLP `mgv_q_delta_encoder(h_state, a)` 产生 delta
4. 构造：

```text
q_state = h_state + delta(h_state, a)
```

5. 再把 `q_state` 送入现有 value head

也就是：

```text
Q(s, a, g) = value_head( h_state + delta(h_state, a), h_goal, h_value )
```

当前会同时训练两种 Q：

- `Q(s, a, g_lang)`
- `Q(s, a, g_img)`

对应目标分别是 chunk-discounted next-value bootstrap：

```text
Q(s, a, g_lang) -> gamma_chunk * stop_gradient( V_target(s', g_lang) )
Q(s, a, g_img)  -> gamma_chunk * stop_gradient( V_target(s', g_img) )
```

其中 `s'` 是 replay 中当前 chunk 执行后的 `next_obs`。

如果 `s' -> g` 的距离足够短，还会先对右边的 `V_target(s', g)` 做 exact override：

```text
V_target(s', g) := gamma_chunk ^ d(s', g)
```

具体规则是：

- `g_img`：只要 `d(s', g_img) <= short_horizon_M_chunk` 就 exact
- `g_lang`：只有 `is_success_sample=True` 且 `d(s', g_lang) <= short_horizon_M_chunk` 才 exact

因此短程时，Q 的 exact 目标实际上变成：

```text
Q_target(s, a, g) = gamma_chunk ^ (1 + d(s', g))
```

当前 Q loss 是：

```text
loss_q = 0.5 * (
    BCEWithLogits(q_lang_logits, q_lang_target)
  + BCEWithLogits(q_img_logits, q_img_target)
)
```

### 2.9 Actor 学习

当前 actor 更新只有两类：`awac` 和 `ddpg_bc`。

#### 2.9.1 AWAC

AWAC 只使用 `g_lang` 信号更新 actor。

先构造：

```text
next_value_target = stop_gradient( sigmoid(V(s'_i, g_lang)) )
next_value = gamma_chunk * next_value_target
```

对 success 样本还会做一层 exact override：

- 若 `d(s'_i, g_goal) = d_ik - 1 <= short_horizon_M_chunk`
- 则把 `next_value_target` 直接替换成 exact `gamma_chunk ^ (d_ik - 1)`

然后根据配置决定 actor signal：

1. `actor_weight_target_mode="advantage"`

```text
signal = next_value - stop_gradient(V(s_i, g_lang))
```

2. `actor_weight_target_mode="next_value"`

```text
signal = next_value
```

再转成权重：

```text
raw_weight = exp(signal / actor_adv_beta)
actor_weight = clamp(raw_weight, max=max_adv_weight)
```

在不同 actor family 下，AWAC 的 supervised target 不同形式如下。

1. `gaussian + awac`

训练目标是 replay 中缓存的 executed action 的 log-prob：

```text
loss_actor = - mean( actor_weight * log pi(a_replay | s_i, g_lang) )
```

这里的 `a_replay` 来自 `forward_inputs["action_for_logprob"]`。

2. `deterministic + awac`

训练目标是 weighted L1：

```text
loss_actor = mean(
    actor_weight * || a_mean(s_i) - a_replay ||_1
)
```

注意这里的 supervision 也是 replay 中缓存的 executed action，而不是额外的 expert action。

#### 2.9.2 DDPG+BC

当 `actor_update_mode="ddpg_bc"` 时，actor 用 `Q(s, a, g_lang)` 更新。

1. actor action

- `gaussian`：用 reparameterized sample
- `deterministic`：直接用 mean action

2. Q-guided actor objective

先预测：

```text
q_actor = sigmoid( Q(s_i, a_actor, g_lang) )
```

再做：

```text
loss_actor_q = - mean(q_actor)
actor_q_weight = ddpg_bc_q_alpha / clamp(mean(q_actor), min=ddpg_bc_q_scale_eps)
loss_actor = actor_q_weight * loss_actor_q
```

实现上会临时冻结 Q 模块参数，只让梯度通过 action 路径回传到 actor，不直接更新 Q 本身。

3. BC regularizer

当前 BC 不是对 replay action，也不是对额外 expert action，而是对 `target_actor_action_model` 的 mean action：

```text
target_mean_action = a_target(s_i)
online_mean_action = a_online(s_i)
```

其中：

- deterministic policy 用 `L1`
- gaussian policy 用 `SmoothL1`

即：

```text
loss_actor_bc = BC( online_mean_action, target_mean_action )
```

最终：

```text
loss_actor = actor_q_weight * loss_actor_q + ddpg_bc_coef * loss_actor_bc
```

因此，当前这条分支更准确地说是：

```text
Q maximization + EMA target-actor regularization
```

而不是经典意义上“对数据集 action 做 BC”的 DDPG+BC。

### 2.10 在线总损失与更新节奏

单个 merged batch 上的总损失是：

```text
total_loss =
    lang_distill_coef * loss_lang_distill
  + transitive_coef * (loss_transitive + loss_lang_transitive)
  + q_loss_coef * loss_q
  + actor_loss_coef * loss_actor
```

为了让 success-only loss 与全 batch loss 在 micro-batch 累积时保持无偏，当前实现又引入了：

- `batch_loss_scale`
- `success_loss_scale`

分别对：

- 全样本都参与的项
- 只有 success 样本参与的项

做缩放。

更新节奏是：

1. replay buffer 达到 `min_buffer_size` 后才开始训练
2. 每个 outer training step 先统计这一步新写入 replay 的 fresh chunk-level transitions 数，记为 `fresh_chunks_this_step`
3. 令一次 optimizer step 消耗的 global replay sample 数为：

```text
replay_samples_per_update_global = actor.global_batch_size
```

4. 当前 off-policy UTD 的定义是：

```text
UTD = 每采集 1 个 fresh chunk-level transition，平均被 replay 复用多少次
```

也就是本实现里：

```text
desired_updates_this_step
    = utd_ratio * fresh_chunks_this_step / replay_samples_per_update_global
```

5. `desired_updates_this_step` 可以是小数，所以 worker 会把它累加到 `utd_update_budget`，再取：

```text
requested_updates_this_step = floor(utd_update_budget)
```

执行完后，剩余小数 budget 留到下一步。
6. 每次 update 都重新从 replay buffer 采一批 merged batch，做一次 optimizer step
7. 本步真实消耗的 replay sample 数为：

```text
replay_samples_consumed_this_step
    = updates_this_step * replay_samples_per_update_global
```

8. 因此当前日志里的真实 UTD 是：

```text
utd_realized
    = replay_samples_consumed_this_step / fresh_chunks_this_step
```

9. step 后 soft update：
   - `target_value_head`
   - `target_actor_action_model`

这里要注意 local / global 的区别：

- `fresh_chunks_this_step_local`：当前 actor rank 本地本步收到的新 chunk 数
- `fresh_chunks_this_step`：所有 actor rank 求和后的 global fresh chunk 数
- `replay_samples_per_update_global`：一次 optimizer step 在全局一共从 replay 取多少条 sample，也就是 `actor.global_batch_size`

因此现在的 `utd_ratio` 已经不是“每轮固定做多少次 update”，而是严格按 fresh on-policy chunk sample 去定义 replay reuse 强度。

### 2.11 当前关键指标

当前最常看的在线指标包括：

- `mgv/total_loss`
- `mgv/lang_distill_loss`
- `mgv/transitive_loss`
- `mgv/lang_transitive_loss`
- `mgv/q_loss`
- `mgv/q_lang_loss`
- `mgv/q_image_loss`
- `mgv/actor_loss`
- `mgv/actor_supervision_l1`
- `mgv/v_lang_s_mean`
- `mgv/v_lang_sp_mean`
- `mgv/lang_target_mean`
- `mgv/transitive_target_mean`
- `mgv/transitive_bridge_target_mean`
- `mgv/transitive_pred_mean`
- `mgv/transitive_abs_error_mean`
- `mgv/lang_transitive_target_mean`
- `mgv/transitive_weight_mean`
- `mgv/transitive_est_distance_mean`
- `mgv/actor_signal_mean` 或 `mgv/adv_mean`
- `mgv/actor_weight_mean`
- `mgv/actor_weight_clip_frac`
- `mgv/q_lang_mean`
- `mgv/q_image_mean`
- `mgv/q_lang_exact_override_rate`
- `mgv/q_image_exact_override_rate`
- `mgv/actor_q_mean`
- `mgv/actor_std_mean`
- `mgv/actor_bc_loss`
- `mgv/actor_bc_target_abs_mean`
- `mgv/success_sample_rate`
- `mgv/chunk_gamma`
- `mgv/short_horizon_M_chunk`
- `mgv/fresh_chunks_this_step`
- `mgv/fresh_chunks_this_step_local`
- `mgv/replay_samples_per_update_global`
- `mgv/desired_updates_this_step`
- `mgv/utd_update_budget`
- `mgv/requested_updates_this_step`
- `mgv/updates_this_step`
- `mgv/replay_samples_consumed_this_step`
- `mgv/utd_realized`
- `mgv/updates_per_env_step`

其中几个特别常用的解释是：

- `mgv/v_lang_s_mean`：当前 `V(s, g_lang)` 的平均概率
- `mgv/v_lang_sp_mean`：当前 `V(s', g_lang)` 的平均概率
- `mgv/transitive_est_distance_mean`：由 `V(s_i, s_k)` 反推出来的平均估计距离
- `mgv/transitive_pred_mean`：当前 `sigmoid(logit_ik)` 的平均值
- `mgv/transitive_abs_error_mean`：当前 `|sigmoid(logit_ik) - target_ik|` 的平均值，其中 `target_ik` 是实际参与监督的 transitive target
- `mgv/actor_supervision_l1`：actor 与其监督目标之间的平均 L1 偏差
- `mgv/success_sample_rate`：当前 merged batch 里 success 样本的比例
- `mgv/q_lang_exact_override_rate` / `mgv/q_image_exact_override_rate`：Q target 里有多少比例走了短程 exact 替换
- `mgv/utd_realized`：这一轮 fresh chunk 实际被 replay 复用了多少次

### 2.12 当前实现边界

当前方法需要明确以下边界：

- 在线 value 形式与离线 MGV 对齐，但整体训练已经是独立的 off-policy RL 过程，不再等价于离线 StarVLA trainer。
- 在线 actor 更新始终围绕 `g_lang` 展开，不使用 image goal 直接更新 actor。
- 在线 Q 学习同时拟合 `g_lang` 与 `g_img`，但 `ddpg_bc` actor 只用 `g_lang` 的 Q。
- `deterministic + awac` 当前监督的是 replay executed action，而不是额外 expert action。
- `ddpg_bc` 的 BC teacher 是 soft-updated target actor，不是 replay 数据标签。

整体上，当前方法可以概括为：

- 离线阶段：在 StarVLA 中学习 image-goal / terminal-goal / language-goal 的 reachability 价值。
- 在线阶段：在 RLinf 中冻结大主干，只更新 action head、MGV head、可选 Q delta encoder 与高斯方差；通过 replay buffer 上的 transitive value 学习、success 条件下的 `g_lang` distill / transitive 学习，以及可选的 `AWAC` 或 `DDPG+BC` actor 更新，提升 language-goal 导向的在线成功率。
