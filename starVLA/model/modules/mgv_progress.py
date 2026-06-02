from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from starVLA.model.modules.vlm.token_utils import ensure_additional_special_tokens


def _feature_std_mean(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2 or x.shape[0] < 2:
        return x.new_zeros(())
    return x.std(dim=0).mean()


def _dist_enabled() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _dist_all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if not _dist_enabled():
        return x
    reduced = x.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def _dist_scalar_mean(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_zeros(())
    total = _dist_all_reduce_sum(x.detach())
    world_size = dist.get_world_size() if _dist_enabled() else 1
    return total / world_size


def _dist_weighted_mean(sum_value: torch.Tensor, count_value: torch.Tensor) -> torch.Tensor:
    global_sum = _dist_all_reduce_sum(sum_value.detach())
    global_count = _dist_all_reduce_sum(count_value.detach())
    return global_sum / global_count.clamp_min(1.0)


def _dist_mean_from_values(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        zero = values.new_zeros(())
        return _dist_weighted_mean(zero, zero)
    return _dist_weighted_mean(values.sum(), values.new_tensor(float(values.numel())))


@dataclass
class MGVDefaultConfig:
    enabled: bool = False
    proj_dim: int = 512
    gamma: float = 0.99
    lambda_mgv: float = 1.0
    use_language_goal: bool = True
    token_init_strategy: str = "normal"
    state_start_token: str = "<state_start>"
    state_end_token: str = "<state_end>"
    goal_img_start_token: str = "<goal_img_start>"
    goal_lang_start_token: str = "<goal_lang_start>"
    goal_img_end_token: str = "<goal_img_end>"
    goal_lang_end_token: str = "<goal_lang_end>"
    value_token: str = "<value>"
    mgv_forward_interval: int = 1
    batch_consistent_action_goal: bool = False
    action_goal_lang_prob: float = 0.8
    eta_distill: float = 0.05
    use_terminal_goal: bool = True
    enable_distill_term_to_lang: bool = True
    enable_distill_lang_to_term: bool = True


class MGVProgressModule(nn.Module):
    def __init__(
        self,
        qwen_vl_interface,
        hidden_size: int,
        mgv_cfg,
        state_instruction_builder: Optional[Callable[[List[str], List[np.ndarray]], List[str]]] = None,
    ) -> None:
        super().__init__()
        self.qwen_vl_interface = qwen_vl_interface
        self.hidden_size = hidden_size
        self.state_instruction_builder = state_instruction_builder

        cfg_dict = dict(MGVDefaultConfig().__dict__)
        if mgv_cfg is not None:
            for key, value in mgv_cfg.items():
                if key in cfg_dict:
                    cfg_dict[key] = value
        self.cfg = type("MGVConfig", (), cfg_dict)()

        token_registration_fn = getattr(self.qwen_vl_interface, "ensure_additional_special_tokens", None)
        if token_registration_fn is None:
            token_mapping = ensure_additional_special_tokens(
                model=self.qwen_vl_interface.model,
                tokenizer=self.qwen_vl_interface.processor.tokenizer,
                tokens=[
                    self.cfg.state_start_token,
                    self.cfg.state_end_token,
                    self.cfg.goal_img_start_token,
                    self.cfg.goal_lang_start_token,
                    self.cfg.goal_img_end_token,
                    self.cfg.goal_lang_end_token,
                    self.cfg.value_token,
                ],
                init_strategy=str(self.cfg.token_init_strategy),
            )
        else:
            token_mapping = token_registration_fn(
                [
                    self.cfg.state_start_token,
                    self.cfg.state_end_token,
                    self.cfg.goal_img_start_token,
                    self.cfg.goal_lang_start_token,
                    self.cfg.goal_img_end_token,
                    self.cfg.goal_lang_end_token,
                    self.cfg.value_token,
                ],
                init_strategy=str(self.cfg.token_init_strategy),
            )
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        self.state_start_token = str(self.cfg.state_start_token)
        self.state_start_token_id = token_mapping[self.state_start_token]
        self.state_end_token = str(self.cfg.state_end_token)
        self.state_end_token_id = token_mapping[self.state_end_token]
        self.goal_img_start_token = str(self.cfg.goal_img_start_token)
        self.goal_img_start_token_id = token_mapping[self.goal_img_start_token]
        self.goal_lang_start_token = str(self.cfg.goal_lang_start_token)
        self.goal_lang_start_token_id = token_mapping[self.goal_lang_start_token]
        self.goal_img_end_token = str(self.cfg.goal_img_end_token)
        self.goal_img_end_token_id = token_mapping[self.goal_img_end_token]
        self.goal_lang_end_token = str(self.cfg.goal_lang_end_token)
        self.goal_lang_end_token_id = token_mapping[self.goal_lang_end_token]
        self.value_token = str(self.cfg.value_token)
        self.value_token_id = token_mapping[self.value_token]
        for token_text, token_id in (
            (self.state_start_token, self.state_start_token_id),
            (self.state_end_token, self.state_end_token_id),
            (self.goal_img_start_token, self.goal_img_start_token_id),
            (self.goal_lang_start_token, self.goal_lang_start_token_id),
            (self.goal_img_end_token, self.goal_img_end_token_id),
            (self.goal_lang_end_token, self.goal_lang_end_token_id),
            (self.value_token, self.value_token_id),
        ):
            token_ids = tokenizer(token_text, add_special_tokens=False)["input_ids"]
            if token_ids != [token_id]:
                raise RuntimeError(
                    f"MGV token {token_text!r} is not encoded as a single special token after registration: {token_ids}"
                )
        self.gamma = float(self.cfg.gamma)
        self.lambda_mgv = float(self.cfg.lambda_mgv)
        self.use_language_goal = bool(self.cfg.use_language_goal)
        self.mgv_forward_interval = max(1, int(getattr(self.cfg, "mgv_forward_interval", 1)))
        self.batch_consistent_action_goal = bool(getattr(self.cfg, "batch_consistent_action_goal", False))
        self.action_goal_lang_prob = float(getattr(self.cfg, "action_goal_lang_prob", 0.8))
        if not 0.0 <= self.action_goal_lang_prob <= 1.0:
            raise ValueError(
                f"MGV requires `action_goal_lang_prob` in [0, 1], got {self.action_goal_lang_prob}."
            )
        self.eta_distill = float(getattr(self.cfg, "eta_distill", 0.05))
        self.enable_distill_term_to_lang = bool(getattr(self.cfg, "enable_distill_term_to_lang", True))
        self.enable_distill_lang_to_term = bool(getattr(self.cfg, "enable_distill_lang_to_term", True))
        self.use_terminal_goal = bool(getattr(self.cfg, "use_terminal_goal", True))

        self.state_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, int(self.cfg.proj_dim)),
        )
        self.img_goal_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, int(self.cfg.proj_dim)),
        )
        self.lang_goal_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, int(self.cfg.proj_dim)),
        )
        self.value_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, int(self.cfg.proj_dim)),
        )
        self.value_projector = nn.Sequential(
            nn.LayerNorm(int(self.cfg.proj_dim) * 3),
            nn.Linear(int(self.cfg.proj_dim) * 3, int(self.cfg.proj_dim)),
            nn.GELU(),
            nn.Linear(int(self.cfg.proj_dim), 1),
        )
    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def _module_device(self) -> torch.device:
        return next(self.value_projector.parameters()).device

    def _module_dtype(self) -> torch.dtype:
        return next(self.value_projector.parameters()).dtype

    def _value_norm_penalty(self, *tensors: Optional[torch.Tensor]) -> torch.Tensor:
        penalty = next(self.value_projector.parameters()).new_zeros(())
        for tensor in tensors:
            if tensor is None or tensor.numel() == 0:
                continue
            penalty = penalty + tensor.norm(dim=-1).pow(2).mean()
        return penalty * 1e-4

    def _build_state_prompt(self, obs_payload: dict) -> str:
        state = obs_payload.get("state", None)
        if state is not None and self.state_instruction_builder is not None:
            prompt = self.state_instruction_builder([""], [state])[0].strip()
            if prompt:
                return f"{self.state_start_token} {prompt} {self.state_end_token}"
        return f"{self.state_start_token} {self.state_end_token}"

    def _build_goal_img_prompt(self) -> str:
        return f"{self.goal_img_start_token} {self.goal_img_end_token}"

    def _build_value_prompt(self) -> str:
        return f"Predict the discounted reachability value from the current observation to the goal. {self.value_token}"

    def _extract_language_goal(self, example: dict) -> Optional[str]:
        text = example.get("language_goal") or example.get("lang")
        if text is None:
            return None
        text = str(text).strip()
        return text if text else None

    def _gather_end_token_hidden(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, token_id: int) -> torch.Tensor:
        batch_size, _, hidden_dim = hidden_states.shape
        index_grid = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        token_mask = input_ids == token_id
        token_counts = token_mask.sum(dim=1)
        if (token_counts < 1).any():
            bad = (token_counts < 1).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(f"Missing required token id {token_id} in samples {bad}")
        token_pos = torch.where(token_mask, index_grid, torch.full_like(index_grid, -1)).max(dim=1).values
        gather_index = token_pos.view(batch_size, 1, 1).expand(batch_size, 1, hidden_dim)
        return hidden_states.gather(dim=1, index=gather_index).squeeze(1)

    def _build_language_prompt(self, language_goal: str) -> str:
        return f"{self.goal_lang_start_token} {language_goal.strip()} {self.goal_lang_end_token}".strip()

    def _build_action_prompt(self, chunk_len: int, action_token: str) -> str:
        action_tokens = action_token * chunk_len
        return f"Please predict the next {chunk_len} robot actions: <action>{action_tokens}<action>."

    def _normalize_image_list(self, images) -> list:
        if isinstance(images, (list, tuple)):
            return list(images)
        return [images]

    def _build_user_batch_inputs(self, batch_contents: List[List[dict]]) -> dict:
        messages = [[{"role": "user", "content": content_items}] for content_items in batch_contents]
        interface_module = type(self.qwen_vl_interface).__module__
        if "QWen2_5" in interface_module:
            from qwen_vl_utils import process_vision_info

            texts = [
                self.qwen_vl_interface.processor.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for message in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            batch_inputs = self.qwen_vl_interface.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            batch_inputs = self.qwen_vl_interface.processor.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        return batch_inputs.to(self.qwen_vl_interface.model.device)

    def _run_content_batch(self, batch_contents: List[List[dict]]) -> tuple[torch.Tensor, dict]:
        qwen_inputs = self._build_user_batch_inputs(batch_contents)
        outputs = self.qwen_vl_interface(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1].float(), qwen_inputs

    def _build_state_content(self, obs_payload: dict) -> List[dict]:
        content = [{"type": "image", "image": image} for image in self._normalize_image_list(obs_payload["image"])]
        content.append({"type": "text", "text": self._build_state_prompt(obs_payload)})
        return content

    def _build_goal_img_content(self, goal_obs: dict) -> List[dict]:
        content = [{"type": "image", "image": image} for image in self._normalize_image_list(goal_obs["image"])]
        content.append({"type": "text", "text": self._build_goal_img_prompt()})
        return content

    def _build_language_goal_content(self, language_goal: str) -> List[dict]:
        return [{"type": "text", "text": self._build_language_prompt(language_goal)}]

    def _build_value_content(self) -> List[dict]:
        return [{"type": "text", "text": self._build_value_prompt()}]

    def _build_action_content(self, chunk_len: int, action_token: str) -> List[dict]:
        return [{"type": "text", "text": self._build_action_prompt(chunk_len, action_token)}]

    def _build_action_sequence_contents(
        self,
        obs_payload: dict,
        chunk_len: int,
        action_token: str,
        language_goal: Optional[str],
        image_goal_obs: Optional[dict],
        use_lang_goal: bool,
    ) -> List[dict]:
        contents = self._build_state_content(obs_payload)
        if use_lang_goal:
            if language_goal is None:
                raise ValueError("Action sequence sampled a language goal but no language goal is available.")
            contents.extend(self._build_language_goal_content(language_goal))
        else:
            if image_goal_obs is None:
                raise ValueError("Action sequence sampled an image goal but no image goal observation is available.")
            contents.extend(self._build_goal_img_content(image_goal_obs))
        contents.extend(self._build_action_content(chunk_len, action_token))
        return contents

    def build_action_training_inputs(
        self,
        examples: List[dict],
        chunk_len: int,
        action_token: str,
    ) -> tuple[dict, torch.Tensor]:
        obs_k = [example["state_obs"] for example in examples]
        image_goal_obs = [example["image_goal_obs"] for example in examples]
        language_goals = [self._extract_language_goal(example) for example in examples]
        device = self._module_device()
        action_goal_is_lang = self._sample_action_goal_is_lang(language_goals, device=device)
        batch_contents = []
        for sample_idx, state_obs in enumerate(obs_k):
            batch_contents.append(
                self._build_action_sequence_contents(
                    obs_payload=state_obs,
                    chunk_len=chunk_len,
                    action_token=action_token,
                    language_goal=language_goals[sample_idx],
                    image_goal_obs=None if image_goal_obs is None else image_goal_obs[sample_idx],
                    use_lang_goal=bool(action_goal_is_lang[sample_idx].item()),
                )
            )
        qwen_inputs = self._build_user_batch_inputs(batch_contents)
        return qwen_inputs, qwen_inputs["input_ids"]

    def build_action_inference_inputs(
        self,
        examples: List[dict],
        chunk_len: int,
        action_token: str,
    ) -> tuple[dict, torch.Tensor]:
        batch_contents = []
        for example in examples:
            language_goal = self._extract_language_goal(example)
            if language_goal is None:
                raise ValueError("Action inference requires a language goal for every sample.")
            obs_payload = {"image": example["image"]}
            if "state" in example:
                obs_payload["state"] = example["state"]
            batch_contents.append(
                self._build_action_sequence_contents(
                    obs_payload=obs_payload,
                    chunk_len=chunk_len,
                    action_token=action_token,
                    language_goal=language_goal,
                    image_goal_obs=None,
                    use_lang_goal=True,
                )
            )
        qwen_inputs = self._build_user_batch_inputs(batch_contents)
        return qwen_inputs, qwen_inputs["input_ids"]

    def _build_value_sequence_contents(
        self,
        obs_payload: dict,
        goal_kind: str,
        goal_obs: Optional[dict] = None,
        language_goal: Optional[str] = None,
    ) -> List[dict]:
        contents = self._build_state_content(obs_payload)
        if goal_kind == "future_image":
            if goal_obs is None:
                raise ValueError("Future-image value sequence requires goal_obs.")
            contents.extend(self._build_goal_img_content(goal_obs))
        elif goal_kind == "terminal_image":
            if goal_obs is None:
                raise ValueError("Terminal-image value sequence requires goal_obs.")
            contents.extend(self._build_goal_img_content(goal_obs))
        elif goal_kind == "lang":
            if language_goal is None:
                raise ValueError("Language value sequence requires a language goal.")
            contents.extend(self._build_language_goal_content(language_goal))
        else:
            raise ValueError(f"Unsupported value goal kind: {goal_kind}")
        contents.extend(self._build_value_content())
        return contents

    def _encode_value_batch(
        self,
        batch_contents: List[List[dict]],
        goal_end_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, qwen_inputs = self._run_content_batch(batch_contents)
        input_ids = qwen_inputs["input_ids"]
        h_state = self._gather_end_token_hidden(hidden, input_ids, self.state_end_token_id)
        h_goal = self._gather_end_token_hidden(hidden, input_ids, goal_end_token_id)
        h_value = self._gather_end_token_hidden(hidden, input_ids, self.value_token_id)
        return h_state, h_goal, h_value

    def _value_head_logit(
        self,
        state_feat: torch.Tensor,
        goal_feat: torch.Tensor,
        value_feat: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([state_feat, goal_feat, value_feat], dim=-1)
        return self.value_projector(fused).squeeze(-1)

    def _build_language_mask(self, examples: List[dict], device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [bool(self._extract_language_goal(example)) for example in examples],
            device=device,
            dtype=torch.bool,
        )

    def _extract_required_metadata_tensor(
        self,
        examples: List[dict],
        key: str,
        device: torch.device,
        *,
        min_value: Optional[int] = 0,
    ) -> torch.Tensor:
        values = []
        for sample_idx, example in enumerate(examples):
            metadata = example.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"MGV requires example[{sample_idx}]['metadata'] to be a dict containing `{key}`."
                )
            if key not in metadata:
                raise ValueError(
                    f"MGV requires example[{sample_idx}]['metadata']['{key}']; no silent fallback is allowed."
                )
            try:
                value = int(metadata[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"MGV requires metadata `{key}` to be integer-like, got {metadata[key]!r} "
                    f"for example[{sample_idx}]."
                ) from exc
            if min_value is not None and value < min_value:
                raise ValueError(
                    f"MGV requires metadata `{key}` >= {min_value}, got {value} for example[{sample_idx}]."
                )
            values.append(value)
        return torch.tensor(values, device=device, dtype=torch.long)

    def _extract_stride_tensor(self, examples: List[dict], device: torch.device) -> torch.Tensor:
        values = []
        for sample_idx, example in enumerate(examples):
            metadata = example.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(
                    "MGV requires example metadata to contain `mgv_temporal_stride`; "
                    f"example[{sample_idx}] has invalid metadata."
                )
            if "mgv_temporal_stride" not in metadata:
                raise ValueError(
                    "MGV requires example metadata `mgv_temporal_stride`; no silent fallback is allowed. "
                    f"Missing at example[{sample_idx}]."
                )
            try:
                value = int(metadata["mgv_temporal_stride"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "MGV requires metadata `mgv_temporal_stride` to be integer-like, "
                    f"got {metadata['mgv_temporal_stride']!r} for example[{sample_idx}]."
                ) from exc
            if value < 1:
                raise ValueError(
                    "MGV requires metadata `mgv_temporal_stride` >= 1, "
                    f"got {value} for example[{sample_idx}]."
                )
            values.append(value)
        return torch.tensor(values, device=device, dtype=torch.long)

    def should_run_mgv(self, current_step: Optional[int]) -> bool:
        if current_step is None:
            return True
        return ((int(current_step) + 1) % self.mgv_forward_interval) == 0

    def _sample_action_goal_is_lang(
        self,
        language_goals: List[Optional[str]],
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = len(language_goals)
        action_goal_is_lang = torch.zeros((batch_size,), device=device, dtype=torch.bool)
        if batch_size < 1:
            return action_goal_is_lang

        has_lang = torch.tensor([goal is not None for goal in language_goals], device=device, dtype=torch.bool)

        if not has_lang.any():
            return action_goal_is_lang

        if self.batch_consistent_action_goal:
            use_lang_goal = bool(torch.rand((), device=device) < self.action_goal_lang_prob)
            if use_lang_goal and bool(has_lang.all()):
                return torch.ones((batch_size,), device=device, dtype=torch.bool)
            return torch.zeros((batch_size,), device=device, dtype=torch.bool)

        sampled = torch.rand(batch_size, device=device) < self.action_goal_lang_prob
        return sampled & has_lang

    def _all_gather_fixed_tensor(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        if not _dist_enabled():
            return [tensor]
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor.detach())
        gathered[dist.get_rank()] = tensor
        return gathered

    def _all_gather_fixed_meta(self, tensor: torch.Tensor) -> torch.Tensor:
        if not _dist_enabled():
            return tensor
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    def _all_gather_variable_tensor(
        self,
        tensor: Optional[torch.Tensor],
        counts: torch.Tensor,
        feature_dim: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        total_count = int(counts.sum().item())
        if total_count == 0:
            return None

        local_count = 0 if tensor is None else int(tensor.shape[0])
        max_count = int(counts.max().item())
        if feature_dim is None:
            if tensor is None:
                raise ValueError("feature_dim must be provided when gathering a variable tensor from a rank with no local rows.")
            feature_dim = int(tensor.shape[1])
        device = self._module_device()
        dtype = self._module_dtype() if tensor is None else tensor.dtype
        padded = torch.zeros((max_count, feature_dim), device=device, dtype=dtype)
        if tensor is not None and local_count > 0:
            padded[:local_count] = tensor

        gathered = self._all_gather_fixed_tensor(padded)
        pieces = []
        for rank_tensor, rank_count in zip(gathered, counts.tolist()):
            rank_count = int(rank_count)
            if rank_count < 1:
                continue
            pieces.append(rank_tensor[:rank_count])
        return torch.cat(pieces, dim=0)

    def _zero_mgv_output(self, zero: torch.Tensor) -> dict:
        return {
            "_loss_mgv_local": zero,
            "loss_mgv": zero,
            "L_mgv": zero,
            "loss_img_reach": zero,
            "L_img_reach": zero,
            "loss_term_reach": zero,
            "L_term_reach": zero,
            "loss_value_norm": zero,
            "L_value_norm": zero,
            "loss_distill": zero,
            "L_distill": zero,
            "img_reach_logit_mean": zero,
            "img_reach_pred_mean": zero,
            "img_reach_target_mean": zero,
            "img_reach_mae": zero,
            "img_reach_rmse": zero,
            "term_reach_logit_mean": zero,
            "term_reach_pred_mean": zero,
            "term_reach_target_mean": zero,
            "term_reach_mae": zero,
            "term_reach_rmse": zero,
        }

    def compute_loss(self, examples: List[dict]) -> dict:
        zero = next(self.value_projector.parameters()).new_zeros(())
        if not self.enabled:
            return self._zero_mgv_output(zero)

        device = self._module_device()
        obs_k = [example["state_obs"] for example in examples]
        image_goal_obs = [example["image_goal_obs"] for example in examples]
        language_goals = [self._extract_language_goal(example) if self.use_language_goal else None for example in examples]
        valid_language_mask = self._build_language_mask(examples, device)
        valid_term_mask = torch.zeros((len(examples),), device=device, dtype=torch.bool)
        terminal_goal_obs = None
        if self.use_terminal_goal:
            terminal_goal_obs = [example["terminal_goal_obs"] for example in examples]
            valid_term_mask = torch.ones((len(examples),), device=device, dtype=torch.bool)

        term_example_indices = torch.nonzero(valid_term_mask, as_tuple=False).flatten()
        h_k_term = None
        h_goal_term = None
        h_value_term = None
        raw_z_goal_term = None
        raw_z_k_term = None
        raw_z_value_term = None
        future_contents = [
            self._build_value_sequence_contents(
                obs_payload=state_obs,
                goal_kind="future_image",
                goal_obs=goal_obs,
            )
            for state_obs, goal_obs in zip(obs_k, image_goal_obs)
        ]
        num_future_rows = len(future_contents)
        image_contents = list(future_contents)
        if terminal_goal_obs is not None and term_example_indices.numel() > 0:
            image_contents.extend(
                self._build_value_sequence_contents(
                    obs_payload=obs_k[idx],
                    goal_kind="terminal_image",
                    goal_obs=terminal_goal_obs[idx],
                )
                for idx in term_example_indices.tolist()
            )
        h_k_img_all, h_goal_img_all, h_value_img_all = self._encode_value_batch(image_contents, self.goal_img_end_token_id)
        h_k_img = h_k_img_all[:num_future_rows]
        h_goal_future = h_goal_img_all[:num_future_rows]
        h_value_img = h_value_img_all[:num_future_rows]
        raw_z_k = self.state_proj(h_k_img)
        raw_z_goal_future = self.img_goal_proj(h_goal_future)
        raw_z_value_img = self.value_proj(h_value_img)
        if terminal_goal_obs is not None and term_example_indices.numel() > 0:
            h_k_term = h_k_img_all[num_future_rows:]
            h_goal_term = h_goal_img_all[num_future_rows:]
            h_value_term = h_value_img_all[num_future_rows:]
            raw_z_k_term = self.state_proj(h_k_term)
            raw_z_goal_term = self.img_goal_proj(h_goal_term)
            raw_z_value_term = self.value_proj(h_value_term)

        lang_example_indices = torch.nonzero(valid_language_mask, as_tuple=False).flatten()
        h_k_lang = None
        h_goal_lang = None
        h_value_lang = None
        raw_z_goal_lang = None
        raw_z_k_lang = None
        raw_z_value_lang = None
        if self.use_language_goal and lang_example_indices.numel() > 0:
            lang_contents = [
                self._build_value_sequence_contents(
                    obs_payload=obs_k[idx],
                    goal_kind="lang",
                    language_goal=language_goals[idx],
                )
                for idx in lang_example_indices.tolist()
            ]
            h_k_lang, h_goal_lang, h_value_lang = self._encode_value_batch(lang_contents, self.goal_lang_end_token_id)
            raw_z_k_lang = self.state_proj(h_k_lang)
            raw_z_goal_lang = self.lang_goal_proj(h_goal_lang)
            raw_z_value_lang = self.value_proj(h_value_lang)

        local_lang_count = torch.tensor([int(lang_example_indices.numel())], device=device, dtype=torch.long)
        global_lang_counts = self._all_gather_fixed_meta(local_lang_count)
        global_h_goal_lang = self._all_gather_variable_tensor(
            h_goal_lang,
            global_lang_counts,
            feature_dim=h_k_img.shape[-1],
        )
        global_z_goal_lang = self._all_gather_variable_tensor(
            raw_z_goal_lang,
            global_lang_counts,
            feature_dim=raw_z_k.shape[-1],
        )

        local_term_count = torch.tensor([int(term_example_indices.numel())], device=device, dtype=torch.long)
        global_term_counts = self._all_gather_fixed_meta(local_term_count)
        global_h_goal_term = self._all_gather_variable_tensor(
            h_goal_term,
            global_term_counts,
            feature_dim=h_k_img.shape[-1],
        )
        global_z_goal_term = self._all_gather_variable_tensor(
            raw_z_goal_term,
            global_term_counts,
            feature_dim=raw_z_k.shape[-1],
        )

        local_k = self._extract_required_metadata_tensor(examples, "k", device=device, min_value=0)
        local_h = self._extract_required_metadata_tensor(examples, "h", device=device, min_value=0)
        local_T = (
            self._extract_required_metadata_tensor(examples, "T", device=device, min_value=0)
            if self.use_terminal_goal
            else torch.zeros((len(examples),), device=device, dtype=torch.long)
        )
        local_stride = self._extract_stride_tensor(examples, device=device)
        if (local_h < local_k).any():
            bad = torch.nonzero(local_h < local_k, as_tuple=False).flatten().tolist()
            raise ValueError(
                f"MGV requires metadata h >= k for every sample; violated at batch indices {bad}."
            )
        if self.use_terminal_goal and (local_T < local_k).any():
            bad = torch.nonzero(local_T < local_k, as_tuple=False).flatten().tolist()
            raise ValueError(
                f"MGV requires metadata T >= k for every sample when terminal goals are enabled; "
                f"violated at batch indices {bad}."
            )
        local_gamma = torch.full(
            (len(examples),),
            fill_value=float(self.gamma),
            device=device,
            dtype=torch.float32,
        ).clamp_(0.0, 1.0)

        global_h_k = torch.cat(self._all_gather_fixed_tensor(h_k_img), dim=0) if _dist_enabled() else h_k_img
        global_h_goal_future = (
            torch.cat(self._all_gather_fixed_tensor(h_goal_future), dim=0) if _dist_enabled() else h_goal_future
        )
        global_h_value = torch.cat(self._all_gather_fixed_tensor(h_value_img), dim=0) if _dist_enabled() else h_value_img
        global_z_k = torch.cat(self._all_gather_fixed_tensor(raw_z_k), dim=0) if _dist_enabled() else raw_z_k
        global_z_goal_future = (
            torch.cat(self._all_gather_fixed_tensor(raw_z_goal_future), dim=0) if _dist_enabled() else raw_z_goal_future
        )
        global_z_value = torch.cat(self._all_gather_fixed_tensor(raw_z_value_img), dim=0) if _dist_enabled() else raw_z_value_img
        global_state_k = self._all_gather_fixed_meta(local_k)
        global_goal_h = self._all_gather_fixed_meta(local_h)
        global_state_stride = self._all_gather_fixed_meta(local_stride)

        local_c_image_delta_raw = (local_h - local_k).float()
        local_c_image_delta_macro = torch.div(
            local_h - local_k,
            local_stride.clamp_min(1),
            rounding_mode="floor",
        ).to(dtype=torch.float32).clamp_min_(0.0)
        c_image_delta_raw = (global_goal_h - global_state_k).float()
        c_image_delta_macro = torch.div(
            global_goal_h - global_state_k,
            global_state_stride.clamp_min(1),
            rounding_mode="floor",
        ).to(dtype=torch.float32).clamp_min_(0.0)

        img_reach_target = torch.pow(local_gamma, local_c_image_delta_macro).to(dtype=raw_z_k.dtype)
        img_reach_logit = self._value_head_logit(raw_z_k, raw_z_goal_future, raw_z_value_img)
        img_reach_pred = torch.sigmoid(img_reach_logit)
        img_reach_error = img_reach_pred - img_reach_target
        loss_img_reach = F.binary_cross_entropy_with_logits(img_reach_logit, img_reach_target)

        term_reach_logit = raw_z_k.new_zeros((0,))
        term_reach_pred = raw_z_k.new_zeros((0,))
        term_reach_target = raw_z_k.new_zeros((0,))
        term_reach_error = raw_z_k.new_zeros((0,))
        loss_term_reach = zero
        if (
            raw_z_goal_term is not None
            and raw_z_k_term is not None
            and raw_z_value_term is not None
            and term_example_indices.numel() > 0
        ):
            local_c_term_delta_macro = torch.div(
                local_T[term_example_indices] - local_k[term_example_indices],
                local_stride[term_example_indices].clamp_min(1),
                rounding_mode="floor",
            ).to(dtype=torch.float32).clamp_min_(0.0)
            term_reach_target = torch.pow(
                local_gamma[term_example_indices],
                local_c_term_delta_macro,
            ).to(dtype=raw_z_k.dtype)
            term_reach_logit = self._value_head_logit(
                raw_z_k_term,
                raw_z_goal_term,
                raw_z_value_term,
            )
            term_reach_pred = torch.sigmoid(term_reach_logit)
            term_reach_error = term_reach_pred - term_reach_target
            loss_term_reach = F.binary_cross_entropy_with_logits(term_reach_logit, term_reach_target)

        loss_distill = zero
        loss_distill_term_to_lang = zero
        loss_distill_lang_to_term = zero
        distill_lang_reach_mean = zero
        distill_term_reach_mean = zero
        distill_lang_reach_logit_mean = zero
        distill_term_reach_logit_mean = zero
        distill_reach_lang_term_mae = zero
        reach_gap_lang_term_l1 = zero
        reach_gap_lang_term_l2 = zero
        corr_reach_lang_term = zero

        joint_lang_term_mask = valid_language_mask & valid_term_mask
        if (
            raw_z_goal_lang is not None
            and raw_z_goal_term is not None
            and raw_z_k_lang is not None
            and raw_z_k_term is not None
            and raw_z_value_lang is not None
            and raw_z_value_term is not None
            and joint_lang_term_mask.any()
        ):
            joint_example_indices = torch.nonzero(joint_lang_term_mask, as_tuple=False).flatten()
            lang_joint_idx = torch.searchsorted(lang_example_indices, joint_example_indices)
            term_joint_idx = torch.searchsorted(term_example_indices, joint_example_indices)

            local_lang_reach_logit = self._value_head_logit(
                raw_z_k_lang[lang_joint_idx],
                raw_z_goal_lang[lang_joint_idx],
                raw_z_value_lang[lang_joint_idx],
            )
            local_term_reach_logit = self._value_head_logit(
                raw_z_k_term[term_joint_idx],
                raw_z_goal_term[term_joint_idx],
                raw_z_value_term[term_joint_idx],
            )
            local_lang_reach = torch.sigmoid(local_lang_reach_logit)
            local_term_reach = torch.sigmoid(local_term_reach_logit)
            distill_lang_reach_mean = _dist_mean_from_values(local_lang_reach)
            distill_term_reach_mean = _dist_mean_from_values(local_term_reach)
            distill_lang_reach_logit_mean = _dist_mean_from_values(local_lang_reach_logit)
            distill_term_reach_logit_mean = _dist_mean_from_values(local_term_reach_logit)

            distill_weight = 0.0
            if self.enable_distill_term_to_lang:
                loss_distill_term_to_lang = F.mse_loss(local_lang_reach_logit, local_term_reach_logit.detach())
                loss_distill = loss_distill + loss_distill_term_to_lang
                distill_weight += 1.0
            if self.enable_distill_lang_to_term:
                loss_distill_lang_to_term = F.mse_loss(local_term_reach_logit, local_lang_reach_logit.detach())
                loss_distill = loss_distill + loss_distill_lang_to_term
                distill_weight += 1.0
            if distill_weight > 1.0:
                loss_distill = loss_distill / distill_weight

            diff = local_lang_reach - local_term_reach
            distill_reach_lang_term_mae = _dist_mean_from_values(diff.abs())
            reach_gap_lang_term_l1 = _dist_mean_from_values(diff.abs())
            reach_gap_lang_term_l2 = torch.sqrt(_dist_mean_from_values(diff.pow(2)))
            if diff.shape[0] > 1:
                centered_lang = local_lang_reach - local_lang_reach.mean()
                centered_term = local_term_reach - local_term_reach.mean()
                corr_num = (centered_lang * centered_term).sum()
                corr_den = (
                    centered_lang.pow(2).sum().clamp_min(1e-8).sqrt()
                    * centered_term.pow(2).sum().clamp_min(1e-8).sqrt()
                )
                corr_reach_lang_term = corr_num / corr_den.clamp_min(1e-8)

        norm_tensors = [raw_z_k, raw_z_goal_future, raw_z_value_img]
        if raw_z_goal_lang is not None:
            norm_tensors.extend([raw_z_goal_lang, raw_z_value_lang])
        if raw_z_goal_term is not None:
            norm_tensors.extend([raw_z_goal_term, raw_z_value_term])
        loss_value_norm = self._value_norm_penalty(*norm_tensors)
        loss_mgv = loss_img_reach + loss_term_reach + self.eta_distill * loss_distill + loss_value_norm

        raw_z_k_norm = raw_z_k.norm(dim=-1)
        raw_z_goal_img_norm = raw_z_goal_future.norm(dim=-1)
        raw_z_goal_lang_norm = raw_z_goal_lang.norm(dim=-1) if raw_z_goal_lang is not None else raw_z_k.new_zeros((0,))
        raw_z_goal_term_norm = raw_z_goal_term.norm(dim=-1) if raw_z_goal_term is not None else raw_z_k.new_zeros((0,))
        raw_z_value_norm = raw_z_value_img.norm(dim=-1)
        raw_z_norm_chunks = [raw_z_k_norm, raw_z_goal_img_norm, raw_z_value_norm]
        if raw_z_goal_lang is not None:
            raw_z_norm_chunks.append(raw_z_goal_lang_norm)
            raw_z_norm_chunks.append(raw_z_value_lang.norm(dim=-1))
        if raw_z_goal_term is not None:
            raw_z_norm_chunks.append(raw_z_goal_term_norm)
            raw_z_norm_chunks.append(raw_z_value_term.norm(dim=-1))
        raw_z_norm_all = torch.cat(raw_z_norm_chunks, dim=0)

        output = {
            "_loss_mgv_local": loss_mgv,
            "loss_mgv": _dist_scalar_mean(loss_mgv),
            "L_mgv": _dist_scalar_mean(loss_mgv),
            "loss_img_reach": _dist_scalar_mean(loss_img_reach),
            "L_img_reach": _dist_scalar_mean(loss_img_reach),
            "loss_term_reach": _dist_scalar_mean(loss_term_reach),
            "L_term_reach": _dist_scalar_mean(loss_term_reach),
            "loss_value_norm": _dist_scalar_mean(loss_value_norm),
            "L_value_norm": _dist_scalar_mean(loss_value_norm),
            "loss_distill": _dist_scalar_mean(loss_distill),
            "L_distill": _dist_scalar_mean(loss_distill),
            "img_reach_logit_mean": _dist_mean_from_values(img_reach_logit),
            "img_reach_pred_mean": _dist_mean_from_values(img_reach_pred),
            "img_reach_target_mean": _dist_mean_from_values(img_reach_target),
            "img_reach_mae": _dist_mean_from_values(img_reach_error.abs()),
            "img_reach_rmse": _dist_mean_from_values(img_reach_error.pow(2)).sqrt(),
            "term_reach_logit_mean": _dist_mean_from_values(term_reach_logit),
            "term_reach_pred_mean": _dist_mean_from_values(term_reach_pred),
            "term_reach_target_mean": _dist_mean_from_values(term_reach_target),
            "term_reach_mae": _dist_mean_from_values(term_reach_error.abs()),
            "term_reach_rmse": _dist_mean_from_values(term_reach_error.pow(2)).sqrt(),
            "distill_lang_reach_mean": distill_lang_reach_mean,
            "distill_term_reach_mean": distill_term_reach_mean,
            "distill_lang_reach_logit_mean": distill_lang_reach_logit_mean,
            "distill_term_reach_logit_mean": distill_term_reach_logit_mean,
            "distill_reach_lang_term_mae": distill_reach_lang_term_mae,
            "reach_gap_lang_term_l1": reach_gap_lang_term_l1,
            "reach_gap_lang_term_l2": reach_gap_lang_term_l2,
            "corr_reach_lang_term": corr_reach_lang_term,
            "eta_distill": zero + self.eta_distill,
            "gamma_mean": _dist_mean_from_values(local_gamma),
            "num_language_goals": zero + int(_dist_all_reduce_sum(valid_language_mask.long().sum()).item()),
            "c_image_distance_min": c_image_delta_raw.min(),
            "c_image_distance_max": c_image_delta_raw.max(),
            "c_image_distance_mean": c_image_delta_raw.mean(),
            "c_image_distance_macro_min": c_image_delta_macro.min(),
            "c_image_distance_macro_max": c_image_delta_macro.max(),
            "c_image_distance_macro_mean": c_image_delta_macro.mean(),
            "h_k_std": _feature_std_mean(global_h_k),
            "h_goal_img_std": _feature_std_mean(global_h_goal_future),
            "h_goal_lang_std": _feature_std_mean(global_h_goal_lang) if global_h_goal_lang is not None else zero,
            "h_goal_term_std": _feature_std_mean(global_h_goal_term) if global_h_goal_term is not None else zero,
            "h_value_std": _feature_std_mean(global_h_value),
            "z_k_std": _feature_std_mean(global_z_k),
            "z_goal_img_std": _feature_std_mean(global_z_goal_future),
            "z_goal_lang_std": _feature_std_mean(global_z_goal_lang) if global_z_goal_lang is not None else zero,
            "z_goal_term_std": _feature_std_mean(global_z_goal_term) if global_z_goal_term is not None else zero,
            "z_value_std": _feature_std_mean(global_z_value),
            "raw_z_norm_mean": _dist_mean_from_values(raw_z_norm_all),
            "raw_z_k_norm_mean": _dist_mean_from_values(raw_z_k_norm),
            "raw_z_goal_img_norm_mean": _dist_mean_from_values(raw_z_goal_img_norm),
            "raw_z_goal_lang_norm_mean": _dist_mean_from_values(raw_z_goal_lang_norm),
            "raw_z_goal_term_norm_mean": _dist_mean_from_values(raw_z_goal_term_norm),
            "raw_z_value_norm_mean": _dist_mean_from_values(raw_z_value_norm),
        }
        if self.enable_distill_term_to_lang:
            output["loss_distill_term_to_lang"] = _dist_scalar_mean(loss_distill_term_to_lang)
            output["L_distill_term_to_lang"] = _dist_scalar_mean(loss_distill_term_to_lang)
        if self.enable_distill_lang_to_term:
            output["loss_distill_lang_to_term"] = _dist_scalar_mean(loss_distill_lang_to_term)
            output["L_distill_lang_to_term"] = _dist_scalar_mean(loss_distill_lang_to_term)
        return output

    @torch.inference_mode()
    def compute_reachability(self, obs_payload: dict, language_goal: str) -> torch.Tensor:
        batch_contents = [
            self._build_value_sequence_contents(
                obs_payload=obs_payload,
                goal_kind="lang",
                language_goal=language_goal,
            )
        ]
        h_state, h_goal_lang, h_value = self._encode_value_batch(batch_contents, self.goal_lang_end_token_id)
        state_feat = self.state_proj(h_state)
        value_feat = self.value_proj(h_value)
        goal_feat = self.lang_goal_proj(h_goal_lang)
        return torch.sigmoid(self._value_head_logit(state_feat, goal_feat, value_feat)).squeeze(0)
