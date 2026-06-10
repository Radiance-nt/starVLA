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

### 2.1 总体目标

在线部分位于 RLinf 仓库中，当前使用的是基于 replay buffer 的 off-policy 版本。核心目标是：

- 保持 StarVLA 的 action 推理路径不变
- 直接复用 StarVLA 离线 MGV 的 value 输入形式和 value head 结构
- 在在线交互数据上做更频繁的 value / actor 更新

当前 worker 在 [fsdp_mgv_trl_policy_worker.py](../../RLinf_offpolicy_launch_20260608/rlinf/workers/actor/fsdp_mgv_trl_policy_worker.py)，StarVLA 适配在 [starvla_action_model.py](../../RLinf_offpolicy_launch_20260608/rlinf/models/embodiment/starvla/starvla_action_model.py)。

### 2.2 在线更新范围

在线阶段只更新两部分：

- action head
- MGV value head

不更新大 VLM 主干。target network 也只为 value head 维护一份轻量副本，用于构造 bootstrap / bridge target，避免复制整套 VLM 主干带来的显存和同步开销。

### 2.3 在线 value 输入与离线保持一致

RLinf 没有重新手写一套 value encoder，而是直接复用 StarVLA 的 `mgv_module`：

- 仍然使用同一套 special tokens
- 仍然使用同一句 value prompt
- 仍然从 `hidden(<state_end>)`、`hidden(<goal_x_end>)`、`hidden(<value>)` 抽特征
- 仍然走 `state_proj / goal_proj / value_proj / value_projector`

当前支持三类在线 value query：

- `goal_kind="future_image"`
- `goal_kind="terminal_image"`
- `goal_kind="lang"`

因此，在线 value 预测与离线 MGV 在输入形式和 head 结构上是一致的。

### 2.4 Replay buffer 与更新节奏

在线训练按下面的循环进行：

1. 用当前 policy 与环境交互。
2. 把轨迹写入 replay buffer。
3. 从 replay buffer 反复采样，做比 on-policy PPO 更频繁的更新。

这套实现是典型的 off-policy, replay-driven 更新方式，UTD 可以大于 1。环境里 success 后仍会继续执行，所以：

- `success_once`：轨迹中任意时刻成功过
- `success_at_end`：轨迹结束时仍成功

这两个指标语义不同，不应混用。

### 2.5 在线 value 学习：Mode A

第一种采样方式来自成功轨迹：

```text
(s_i, s_success, g_lang)
```

目标是让：

```text
V(s_i, g_lang)
```

去学习：

```text
V(s_i, s_success)
```

其中：

- 如果距离足够短，直接用 exact target `gamma^K`
- 如果距离超过 short horizon，就用 target value head 对 `future_image` 成功状态做预测

对应 loss 仍然是：

```text
BCEWithLogits(logit_lang, target)
```

这部分可以理解为在线版的 `lang <- success-image` 蒸馏。

### 2.6 在线 value 学习：Mode B

第二种采样方式来自任意轨迹：

```text
(s_i, s_j, s_k, s_{i+1}, g_lang)
```

这里主要训练 future-image reachability：

```text
V(s_i, s_k)
```

目标来自 transitive bridge：

```text
target_ik = implicit_max( V(s_i, s_j) * V(s_j, s_k) )
```

当前实现里：

- 短程仍然保留 exact `gamma^K`
- 非短程部分用 `bridge_target = target_ij * target_jk`
- 再用 expectile-weighted BCE 作为 implicit-max surrogate

也就是它不是直接对 `bridge_target` 做普通 BCE，而是通过 expectile 加权，让较大的 bridge target 在优化时权重更高，从而近似 TRL 式的 implicit max。

此外还保留了 distance reweighting。做法是：

1. 先从 `sigmoid(logit_ik)` 反推出估计距离 `d_hat`
2. 再按 `1 / (1 + d_hat)^lambda` 给 transitive loss 加权

这样可以减弱极长距离样本对优化的主导。

### 2.7 在线 actor 学习

Actor 不是走 PPO ratio / clip 形式，而是 advantage-weighted BC。当前 advantage 定义为：

```text
A = gamma_chunk * V_lang(s') - V_lang(s)
```

其中：

- `s` 是当前 coarse state
- `s'` 是执行完当前 action chunk 后的下一 coarse state

再把它变成权重：

```text
actor_weight = exp(A / beta)
```

并做上界截断。最后用这个权重去加权行为策略动作的 log-prob：

```text
loss_actor = - actor_weight * log pi(a | s, g_lang)
```

因此，在线 actor 学习本质上是“value-guided 的加权行为克隆”，而不是标准 on-policy PPO。

### 2.8 在线总 loss

当前在线总 loss 为：

```text
loss =
    lang_distill_coef * loss_lang_distill
  + transitive_coef * loss_transitive
  + actor_loss_coef * loss_actor
```

也就是：

- 一项 lang distill
- 一项 transitive value 学习
- 一项 actor 的 advantage-weighted BC

### 2.9 在线关键指标

常看的指标包括：

- `mgv/lang_distill_loss`
- `mgv/transitive_loss`
- `mgv/actor_loss`
- `mgv/lang_target_mean`
- `mgv/transitive_target_mean`
- `mgv/transitive_bridge_target_mean`
- `mgv/transitive_weight_mean`
- `mgv/transitive_est_distance_mean`
- `mgv/adv_mean`
- `mgv/adv_std`
- `mgv/actor_weight_mean`
- `mgv/chunk_gamma`
- `mgv/short_horizon_M_chunk`
- `mgv/short_horizon_M_primitive`
- `success_once`
- `success_at_end`

其中要特别注意：

- `mgv/adv_std` 目前是在 micro-batch 内计算的；如果 `micro_batch_size=1`，它会稳定显示为 `0`，这不代表真实 advantage 没有方差，只是这个 metric 在该设置下没有解释力。
- `success_once` 高而 `success_at_end` 低，在“成功后环境仍继续执行”的设定下是可能正常出现的。

### 2.10 当前实现里的边界

当前在线实现还有几个需要明确的边界：

- `lang_transitive_target` 目前只记录 metric，不进入 loss。
- 在线 actor 默认只按 `lang goal` 学习，不把 image goal 混入 policy 更新。
- 在线 value 形式已经和离线 StarVLA MGV 对齐，但它仍然是一个独立的 off-policy 训练过程，不再等价于原始 PPO。

整体上，可以把现在的方法概括为：

- 离线阶段：在 StarVLA 上训练 image / terminal / language 三类 reachability 价值。
- 在线阶段：在 RLinf 中复用这套 value 形式，用 off-policy replay 更新 actor head 与 value head，让价值学习和策略学习都围绕 language-goal 成功率展开。
