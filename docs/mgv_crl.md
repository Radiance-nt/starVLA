# MGV 方法文档

这份文档只描述当前分支里保留的 MGV 实现，不再记录已经删除的兼容路径。

## 1. 整体结构

当前训练显式拆成两个 forward：

- `forward_mode="action"`：每个 step 都执行，负责动作学习。
- `forward_mode="mgv"`：只在满足 `mgv_forward_interval` 时执行，负责可达性价值学习。

训练循环位于 [train_starvla.py](/mnt/hwfile/linzhanhui/projects/starVLA/starVLA/training/train_starvla.py)。主路径固定是：

1. action forward
2. 如果到了设定频率，再跑一次 mgv forward
3. 两个 loss 分别 backward 到同一个模型

没有旧的 `combined` 路径，也没有 action/mgv 混在一次 forward 里的隐藏分支。

## 2. 特殊 token

MGV 额外注册的 special token 定义在 [mgv_progress.py](/mnt/hwfile/linzhanhui/projects/starVLA/starVLA/model/modules/mgv_progress.py)：

- `<state_start>`
- `<state_end>`
- `<goal_img_start>`
- `<goal_img_end>`
- `<goal_lang_start>`
- `<goal_lang_end>`
- `<value>`

这些 token 会作为 tokenizer 的 `additional_special_tokens` 新注册，并在模型 embedding 表里新增可训练行。

## 3. Action 分支

### 3.1 输入形式

action 分支只会二选一使用 goal：

- image-goal action:

```text
[state images] + "<state_start> ... <state_end>" +
[future goal images] + "<goal_img_start> <goal_img_end>" +
"Please predict the next N robot actions: <action>🔍...🔍<action>."
```

- lang-goal action:

```text
[state images] + "<state_start> ... <state_end>" +
"<goal_lang_start> {language goal} <goal_lang_end>" +
"Please predict the next N robot actions: <action>🔍...🔍<action>."
```

`terminal_goal_obs` 不进入 action 分支。

### 3.2 采样语义

action goal 的混合由：

- `action_goal_lang_prob`
- `batch_consistent_action_goal`

控制。

含义是：

- `action_goal_lang_prob = 1.0`：action 永远只看 language goal。
- `action_goal_lang_prob = 0.0`：action 永远只看 future image goal。
- 中间值：按概率采样。

如果 `batch_consistent_action_goal=true`，整批样本共享一次 Bernoulli 采样；否则逐样本采样。

### 3.3 推理语义

`predict_action()` 固定走 lang-goal action inference。也就是说，推理时输入是：

```text
[state] + [language goal] + [action prompt]
```

这和当前主实验里“policy 只用 lang goal”的部署语义一致。

## 4. MGV 分支

### 4.1 MGV 输入形式

MGV forward 不预测 action，而是单独构造 value query。value instruction 固定为：

```text
Predict the discounted reachability value from the current observation to the goal. <value>
```

然后分别构造三类目标：

- future image goal
- terminal image goal
- language goal

对应的序列分别是：

```text
[state] + [future image goal] + [value prompt]
[state] + [terminal image goal] + [value prompt]
[state] + [language goal] + [value prompt]
```

三类 goal 是分别前向编码的，不会互相出现在同一个输入序列里。

### 4.2 Value head

MGV 使用两级投影：

1. token-level projector
2. final value projector

具体地，会抽取：

- `hidden(<state_end>)`
- `hidden(<goal_img_end>)` 或 `hidden(<goal_lang_end>)`
- `hidden(<value>)`

然后过下面这些模块：

- `state_proj`: `LayerNorm -> Linear -> GELU -> Linear`
- `img_goal_proj`: `LayerNorm -> Linear -> GELU -> Linear`
- `lang_goal_proj`: `LayerNorm -> Linear -> GELU -> Linear`
- `value_proj`: `LayerNorm -> Linear -> GELU -> Linear`

最后把三路投影结果拼接，再经过：

- `value_projector`: `LayerNorm -> Linear -> GELU -> Linear -> scalar logit`

输出一个 value logit，随后取 `sigmoid(logit)` 得到 `[0, 1]` 范围的 reachability prediction。

## 5. 监督目标

### 5.1 Image / terminal reachability

当前监督目标是折扣 reachability：

```text
target = gamma ^ K
```

其中：

- future image goal: `K = floor((h - k) / stride)`
- terminal goal: `K = floor((T - k) / stride)`

这里的 `k / h / T / mgv_temporal_stride` 都必须由数据集 metadata 显式提供；当前实现不再允许 silent fallback。

loss 是：

```text
BCEWithLogits(logit, target)
```

### 5.2 Lang-term distill

如果同时有 `g_lang` 和 `g_term`，则额外计算：

- `logit_lang`
- `logit_term`

然后做 teacher-probability distillation。默认保留的主方向是：

```text
BCEWithLogits(
    logit_lang,
    sigmoid(stop_gradient(logit_term))
)
```

也就是说：

- student 仍然输出 `logit_lang`
- teacher 侧不再直接在 logit 空间做 MSE
- 而是先把 `logit_term` 经过 `sigmoid` 变成 `[0, 1]` 的 soft reachability target，再喂给 `BCEWithLogits`

如果打开反向 distill，则对应地使用：

```text
BCEWithLogits(
    logit_term,
    sigmoid(stop_gradient(logit_lang))
)
```

是否启用由：

- `enable_distill_term_to_lang`
- `enable_distill_lang_to_term`

控制；总权重由 `eta_distill` 控制。

### 5.3 Norm regularization

当前实现仍保留一项轻量 norm penalty：

```text
1e-4 * mean(||z||^2)
```

它加在参与 MGV 的投影向量上，用来抑制 value 表征范数无约束增大。

## 6. Loss 汇总

MGV 局部 loss 为：

```text
loss_mgv =
    loss_img_reach
  + loss_term_reach
  + eta_distill * loss_distill
  + loss_value_norm
```

trainer 侧最终再乘：

```text
lambda_mgv
```

并与 action loss 共同优化。

## 7. 关键 metric

最常看的指标有：

- `L_act`: action L1 loss
- `L_mgv`: 总 MGV loss
- `L_img_reach`: future image reachability BCE
- `L_term_reach`: terminal reachability BCE
- `L_distill`: lang/term distill loss
- `L_value_norm`: value 表征 norm 正则
- `img_reach_pred_mean`: future image reachability 预测均值
- `img_reach_target_mean`: future image reachability 目标均值
- `term_reach_pred_mean`: terminal reachability 预测均值
- `term_reach_target_mean`: terminal reachability 目标均值
- `distill_lang_reach_mean`: lang branch 的平均 reachability 预测
- `distill_term_reach_mean`: term branch 的平均 reachability 预测
- `distill_reach_lang_term_mae`: lang/term reachability 的平均绝对差
- `raw_z_*_norm_mean`: 各类 MGV 投影向量的范数统计

## 8. 当前保留的配置项

MGV 相关的最小配置接口现在是：

```yaml
framework:
  mgv:
    enabled: true
    proj_dim: 512
    gamma: 0.99
    lambda_mgv: 1.0
    use_language_goal: true
    token_init_strategy: normal
    state_start_token: <state_start>
    state_end_token: <state_end>
    goal_img_start_token: <goal_img_start>
    goal_lang_start_token: <goal_lang_start>
    goal_img_end_token: <goal_img_end>
    goal_lang_end_token: <goal_lang_end>
    value_token: <value>
    mgv_forward_interval: 5
    batch_consistent_action_goal: true
    action_goal_lang_prob: 0.8
    eta_distill: 0.05
    use_terminal_goal: true
    enable_distill_term_to_lang: true
    enable_distill_lang_to_term: false
```

这份分支已经删除了旧的 start-token 兼容入口、`combined` forward、lang temporal diagnostic 开关，以及旧的 terminal alignment 兼容键。文档和代码现在只保留当前真实在用的实现。
