# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-OFT Framework

A lightweight implementation that uses an action special token to parallelly predict continuous actions
conditioned on multi-view images plus a language instruction (shares parameters with the VLM).
Inspired by OpenVLA-OFT
Key Points:
  - Qwen2.5 vision-language backbone
  - Injects an action special token into the VLM
  - Continuous action prediction via L1 regression over the action special token hidden states


Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
  or /starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md （adpat a little code)

"""

import inspect
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
SAFE_FUTURE_TOKEN_CHARS = ["🛸", "🔭", "🧭", "🧪", "🧠", "🎯", "🎲", "🎮"]

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import add_discretized_state_to_instruction, merge_framework_config
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.training.trainer_utils.trainer_tools import resize_images


def _select_attention_heads(hidden_size: int) -> int:
    for num_heads in (8, 4, 2, 1):
        if hidden_size % num_heads == 0:
            return num_heads
    return 1


class PMAPooling(nn.Module):
    """Single-query PMA-style pooling over a token set."""

    def __init__(self, hidden_size: int, num_heads: int, ff_hidden_size: Optional[int] = None) -> None:
        super().__init__()
        ff_hidden_size = ff_hidden_size or hidden_size * 2
        self.token_ff = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, ff_hidden_size),
            nn.GELU(),
            nn.Linear(ff_hidden_size, hidden_size),
        )
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.out_ff = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        query: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        refined_tokens = self.token_ff(tokens)
        key_padding_mask = None
        if token_mask is not None:
            key_padding_mask = ~token_mask.bool()
        attn_out, _ = self.attn(
            query=self.query_norm(query),
            key=self.key_norm(refined_tokens),
            value=refined_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.out_ff(attn_out + query)


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenOFT
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenOFTDefaultConfig:
    """QwenOFT framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier (must match @FRAMEWORK_REGISTRY.register) ---
    name: str = "QwenOFT"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
        }
    )

    # === Action head (MLP regression over action special tokens) ===
    action_model: dict = field(
        default_factory=lambda: {
            # Action head architecture type
            "action_model_type": "MLP",
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # Hidden dim for the action MLP (auto-set from VLM hidden_size at runtime)
            "action_hidden_dim": 2560,
            # How many future steps to predict
            "future_action_window_size": 8,
            # How many past steps included in action chunk (usually 0)
            "past_action_window_size": 0,
        }
    )

    use_future_tokens: bool = False
    num_future_tokens: int = 4
    contrastive_dim: int = 256
    dfc_temperature: float = 0.1
    dfc_gamma: float = 0.99
    lambda_dfc: float = 0.05
    target_encoder_type: str = "stopgrad"
    use_same_language_negatives: bool = False


@FRAMEWORK_REGISTRY.register("QwenOFT")
class Qwenvl_OFT(baseframework):
    """
    Multimodal vision-language-action model (OFT variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Action special token injected into the VLM sequence
      - MLP regression head over action token hidden states (L1 loss)

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenOFTDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align action_hidden_dim to VLM hidden_size at runtime
        self.config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)
        self.hidden_size = self.qwen_vl_interface.model.config.hidden_size

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon
        # self.hidden_dim = config.framework.action_model.action_hidden_dim

        self.action_token = "🔍"  # TODO also can add spacail token to Qwen, but too complex
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]

        # L1 loss
        self.l1_loss = nn.L1Loss()

        legacy_future_cfg = self.config.framework.get("future_tokens", {})
        legacy_contrastive_cfg = self.config.framework.get("contrastive", {})

        self.future_tokens_enabled = bool(
            self.config.framework.get("use_future_tokens", legacy_future_cfg.get("enabled", False))
        )
        self.num_future_tokens = int(self.config.framework.get("num_future_tokens", len(legacy_future_cfg.get("horizons", [])) or 4))
        self.contrastive_dim = int(
            self.config.framework.get(
                "contrastive_dim",
                legacy_contrastive_cfg.get("projector_out_dim", 256),
            )
        )
        self.dfc_temperature = float(
            self.config.framework.get("dfc_temperature", legacy_contrastive_cfg.get("temperature", 0.1))
        )
        self.dfc_gamma = float(self.config.framework.get("dfc_gamma", 0.99))
        self.lambda_dfc = float(
            self.config.framework.get("lambda_dfc", legacy_contrastive_cfg.get("loss_weight", 0.05))
        )
        self.target_encoder_type = str(self.config.framework.get("target_encoder_type", "stopgrad")).lower()
        self.use_same_language_negatives = bool(self.config.framework.get("use_same_language_negatives", False))

        if self.target_encoder_type != "stopgrad":
            raise NotImplementedError(
                f"Unsupported target_encoder_type={self.target_encoder_type!r}; only 'stopgrad' is implemented."
            )

        self.future_token_chars: list[str] = []
        self.future_token_ids: list[int] = []
        self.future_token_parameters = None
        self.future_condition_fuser = None
        self.anchor_pma = None
        self.anchor_seed = None
        self.anchor_projector = None
        self.target_pma = None
        self.target_query_seed = None
        self.target_language_proj = None
        self.target_projector = None

        if self.future_tokens_enabled:
            if self.num_future_tokens > len(SAFE_FUTURE_TOKEN_CHARS):
                raise ValueError(
                    f"Configured {self.num_future_tokens} future tokens, but only "
                    f"{len(SAFE_FUTURE_TOKEN_CHARS)} verified single-token markers are available."
                )

            self.future_token_chars = SAFE_FUTURE_TOKEN_CHARS[: self.num_future_tokens]
            for token_char in self.future_token_chars:
                token_ids = self.qwen_vl_interface.processor.tokenizer(
                    token_char, add_special_tokens=False
                )["input_ids"]
                if len(token_ids) != 1:
                    raise ValueError(f"Future marker {token_char!r} must tokenize to exactly one token, got {token_ids}")
                self.future_token_ids.append(token_ids[0])

            pma_heads = _select_attention_heads(self.hidden_size)
            self.future_token_parameters = nn.Parameter(
                torch.randn(self.num_future_tokens, self.hidden_size) * 0.02
            )
            self.future_condition_fuser = nn.Sequential(
                nn.LayerNorm(self.hidden_size * 2),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            self.anchor_pma = PMAPooling(self.hidden_size, num_heads=pma_heads)
            self.anchor_seed = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
            self.anchor_projector = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.contrastive_dim),
            )
            self.target_pma = PMAPooling(self.hidden_size, num_heads=pma_heads)
            self.target_query_seed = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
            self.target_language_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.target_projector = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.contrastive_dim),
            )

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward: directly regress future actions (no diffusion).

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # List[ndarray (1, state_dim)] or None

        base_instructions = self._build_base_instructions(instructions, state)
        model_instructions = self._append_action_and_future_prompt(base_instructions)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=model_instructions)
        input_ids = qwen_inputs.get("input_ids", None)
        model_inputs = self._prepare_model_inputs(qwen_inputs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **model_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action token embeddings as action prediction queries
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]
            future_hidden = None
            if self.future_tokens_enabled:
                future_hidden = self._gather_specific_token_embeddings(
                    last_hidden, input_ids, self.future_token_ids
                )

            pred_actions = self._predict_action(action_queries, future_hidden=future_hidden)  # (B, chunk_len, action_dim)

            # Label alignment: take the last chunk_len segment
            actions = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            # Compute L1 loss
            action_loss = self.l1_loss(pred_actions, actions_target)

            total_loss = action_loss
            output_dict = {"action_loss": action_loss, "L_act": action_loss}

            if self.future_tokens_enabled:
                dfc_metrics = self._compute_dfc_loss(
                    examples=examples,
                    base_instructions=base_instructions,
                    future_hidden=future_hidden,
                )
                total_loss = total_loss + self.lambda_dfc * dfc_metrics["L_dfc"]
                output_dict.update(dfc_metrics)

            output_dict["total_loss"] = total_loss

        return output_dict

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # List[ndarray (1, state_dim)] or None

        base_instructions = self._build_base_instructions(instructions, state)

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        model_instructions = self._append_action_and_future_prompt(base_instructions)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=model_instructions)
        input_ids = qwen_inputs.get("input_ids", None)
        model_inputs = self._prepare_model_inputs(qwen_inputs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **model_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action token embeddings as action prediction queries
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]
            future_hidden = None
            if self.future_tokens_enabled:
                future_hidden = self._gather_specific_token_embeddings(
                    last_hidden, input_ids, self.future_token_ids
                )
            pred_actions = self._predict_action(action_queries, future_hidden=future_hidden)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,  # [B, L, H]
        input_ids: torch.Tensor,  # [B, L]
        action_token_id=None,  # Can be int or List[int]
    ) -> torch.Tensor:
        """
        Vectorized batch extraction of action token embeddings:
          - No per-sample for loop
          - Select the last chunk_len action placeholder tokens from each sample
        Args:
            last_hidden: [B, L, H]
            input_ids:   [B, L]
            action_token_id: int or List[int]
        Returns:
            action_queries: [B, chunk_len, H]
        """
        if action_token_id is None:
            raise ValueError("action_token_id must not be None")

        device = input_ids.device
        B, L, H = last_hidden.shape

        # Support multiple ids (e.g., multiple variants)
        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            # torch.isin requires PyTorch >=1.10
            mask = torch.isin(input_ids, id_list)
        else:
            mask = input_ids == action_token_id  # [B, L]

        counts = mask.sum(dim=1)  # [B]
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"The following samples have insufficient action tokens (< {self.chunk_len}): {insufficient} |"
                f" counts={counts.tolist()}"
            )

        # Position indices
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))  # Set non-action positions to -1

        # Take the last chunk_len positions (higher indices = later in sequence)
        # Note: count sufficiency already verified, so -1 won't be incorrectly selected
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values  # [B, chunk_len] unsorted
        # Sort in temporal order
        selected_pos = topk_pos.sort(dim=-1).values  # [B, chunk_len]

        # Gather
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)  # [B, chunk_len, H]
        action_queries = last_hidden.gather(dim=1, index=expanded_index)  # [B, chunk_len, H]
        return action_queries

    def _build_base_instructions(self, instructions: List[str], state: Optional[List[np.ndarray]]) -> List[str]:
        return self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions

    def _append_action_and_future_prompt(self, instructions: List[str]) -> List[str]:
        future_prompt = ""
        if self.future_tokens_enabled and self.future_token_chars:
            marker_text = "".join(f"[{token_char}]" for token_char in self.future_token_chars)
            future_prompt = f" Intention tokens: {marker_text}."

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = (
            f"{future_prompt} Please predict the next {self.chunk_len} robot actions: "
            f"<action>{action_tokens}<action>."
        )
        return [instruction + prompt_suffix for instruction in instructions]

    def _gather_specific_token_embeddings(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        token_ids: List[int],
    ) -> torch.Tensor:
        """Gather one embedding per configured marker token in the provided order."""
        device = input_ids.device
        batch_size, _, hidden_dim = last_hidden.shape
        positions = []

        index_grid = torch.arange(input_ids.shape[1], device=device).unsqueeze(0).expand_as(input_ids)
        for token_id in token_ids:
            mask = input_ids == token_id
            counts = mask.sum(dim=1)
            if (counts < 1).any():
                bad_indices = (counts < 1).nonzero(as_tuple=False).flatten().tolist()
                raise RuntimeError(f"Missing future marker token {token_id} in batch samples {bad_indices}")
            token_pos = torch.where(mask, index_grid, torch.full_like(index_grid, -1)).max(dim=1).values
            positions.append(token_pos)

        stacked_positions = torch.stack(positions, dim=1)
        gather_index = stacked_positions.unsqueeze(-1).expand(batch_size, len(token_ids), hidden_dim)
        return last_hidden.gather(dim=1, index=gather_index)

    def _condition_action_queries(
        self,
        action_queries: torch.Tensor,
        future_hidden: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if future_hidden is None or self.future_condition_fuser is None:
            return action_queries

        future_context = self._pool_future_tokens(future_hidden).unsqueeze(1).expand(-1, action_queries.shape[1], -1)
        fused_input = torch.cat([action_queries, future_context], dim=-1)
        return self.future_condition_fuser(fused_input)

    def _prepare_model_inputs(self, qwen_inputs: dict) -> dict:
        if not self.future_tokens_enabled or self.future_token_parameters is None:
            return qwen_inputs

        input_ids = qwen_inputs.get("input_ids", None)
        if input_ids is None:
            raise ValueError("Future token conditioning requires `input_ids` in Qwen inputs.")

        inputs_embeds = self.qwen_vl_interface.model.get_input_embeddings()(input_ids)
        for token_index, token_id in enumerate(self.future_token_ids):
            token_mask = input_ids == token_id
            if not token_mask.any():
                raise RuntimeError(f"Future marker token {token_id} missing from the input sequence.")
            replacement = self.future_token_parameters[token_index].to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            inputs_embeds = torch.where(
                token_mask.unsqueeze(-1),
                replacement.view(1, 1, -1),
                inputs_embeds,
            )

        model_inputs = dict(qwen_inputs)
        model_inputs.pop("input_ids", None)
        model_inputs["inputs_embeds"] = inputs_embeds
        return model_inputs

    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.to(hidden_states.device).unsqueeze(-1).to(hidden_states.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    def _pool_future_tokens(self, future_hidden: torch.Tensor) -> torch.Tensor:
        if self.anchor_pma is None or self.anchor_seed is None:
            raise ValueError("Future token pooling requested but anchor PMA is not initialized.")

        seed = self.anchor_seed.expand(future_hidden.shape[0], -1, -1).to(device=future_hidden.device, dtype=future_hidden.dtype)
        pooled = self.anchor_pma(future_hidden, seed)
        return pooled[:, 0, :]

    def _predict_action(
        self,
        action_queries: torch.Tensor,
        future_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        predict_action = getattr(self.action_model, "predict_action")
        signature = inspect.signature(predict_action)
        if future_hidden is not None and "future_condition_tokens" in signature.parameters:
            return predict_action(action_queries, future_condition_tokens=future_hidden)
        if future_hidden is not None:
            action_queries = self._condition_action_queries(action_queries, future_hidden)
        return predict_action(action_queries)

    def _build_future_batch(
        self,
        examples: List[dict],
    ) -> tuple[List[List[Image.Image]], torch.Tensor]:
        future_images = []
        sampled_deltas = []
        for example in examples:
            future_image = example.get("future_image", None)
            if future_image is None:
                raise ValueError(
                    "DFT-VLA requires `future_image` in each batch sample. "
                    "Set datasets.vla_data.return_future_obs=true."
                )
            future_images.append(future_image)
            sampled_deltas.append(int(example.get("future_delta", 0)))
        return future_images, torch.tensor(sampled_deltas, device=self.qwen_vl_interface.model.device, dtype=torch.float32)

    def _encode_text_only_targets(
        self,
        instructions: List[str],
    ) -> torch.Tensor:
        text_only_images = [[] for _ in instructions]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=text_only_images, instructions=instructions)
        with torch.no_grad():
            text_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        return self._pool_hidden_states(text_outputs.hidden_states[-1].float(), qwen_inputs.get("attention_mask", None))

    def _encode_future_visual_tokens(
        self,
        future_images: List[List[Image.Image]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        empty_instructions = [""] * len(future_images)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=future_images, instructions=empty_instructions)
        pixel_values = qwen_inputs.get("pixel_values", None)
        image_grid_thw = qwen_inputs.get("image_grid_thw", None)
        if pixel_values is None or image_grid_thw is None:
            raise ValueError("Future target encoding requires pixel_values and image_grid_thw.")

        with torch.no_grad():
            image_features = self.qwen_vl_interface.model.get_image_features(pixel_values, image_grid_thw)

        if isinstance(image_features, tuple) and len(image_features) == 2 and isinstance(image_features[0], (list, tuple)):
            image_feature_chunks = image_features[0]
        else:
            image_feature_chunks = image_features

        view_counts = [len(sample_images) for sample_images in future_images]
        grouped_tokens = []
        max_tokens = 0
        cursor = 0
        for view_count in view_counts:
            sample_chunks = image_feature_chunks[cursor : cursor + view_count]
            cursor += view_count
            sample_tokens = torch.cat(sample_chunks, dim=0)
            grouped_tokens.append(sample_tokens)
            max_tokens = max(max_tokens, sample_tokens.shape[0])

        batch_size = len(grouped_tokens)
        hidden_size = grouped_tokens[0].shape[-1]
        token_tensor = grouped_tokens[0].new_zeros((batch_size, max_tokens, hidden_size))
        token_mask = torch.zeros((batch_size, max_tokens), device=grouped_tokens[0].device, dtype=torch.bool)
        for idx, sample_tokens in enumerate(grouped_tokens):
            token_tensor[idx, : sample_tokens.shape[0]] = sample_tokens
            token_mask[idx, : sample_tokens.shape[0]] = True
        return token_tensor, token_mask

    def _encode_future_targets(
        self,
        examples: List[dict],
        base_instructions: List[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.target_pma is None or self.target_projector is None or self.target_query_seed is None:
            raise ValueError("Future target encoder requested but DFT target modules are not initialized.")

        future_images, sampled_deltas = self._build_future_batch(examples)
        visual_tokens, visual_mask = self._encode_future_visual_tokens(future_images)
        language_features = self._encode_text_only_targets(base_instructions)
        target_query = self.target_query_seed.expand(visual_tokens.shape[0], -1, -1)
        target_query = target_query + self.target_language_proj(language_features).unsqueeze(1)
        pooled_future = self.target_pma(visual_tokens.float(), target_query, token_mask=visual_mask)[:, 0, :]
        target_repr = F.normalize(self.target_projector(pooled_future), dim=-1)
        return target_repr, sampled_deltas

    def _compute_dfc_loss(
        self,
        examples: List[dict],
        base_instructions: List[str],
        future_hidden: Optional[torch.Tensor],
    ) -> dict:
        if future_hidden is None or self.anchor_projector is None:
            raise ValueError("DFT-VLA loss requested but future token branch is not initialized.")

        pooled_future = self._pool_future_tokens(future_hidden.float())
        anchors = F.normalize(self.anchor_projector(pooled_future), dim=-1)
        targets, sampled_deltas = self._encode_future_targets(examples, base_instructions)

        gathered_targets = targets
        gathered_deltas = sampled_deltas
        rank = 0
        label_offset = 0
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            local_batch_size = torch.tensor([targets.shape[0]], device=targets.device, dtype=torch.long)
            batch_size_list = [torch.zeros_like(local_batch_size) for _ in range(world_size)]
            dist.all_gather(batch_size_list, local_batch_size)
            label_offset = int(torch.stack(batch_size_list[:rank]).sum().item()) if rank > 0 else 0
            target_list = [torch.zeros_like(targets) for _ in range(world_size)]
            delta_list = [torch.zeros_like(sampled_deltas) for _ in range(world_size)]
            dist.all_gather(target_list, targets.detach())
            dist.all_gather(delta_list, sampled_deltas.detach())
            gathered_targets = torch.cat(target_list, dim=0)
            gathered_deltas = torch.cat(delta_list, dim=0)

        logits = torch.matmul(anchors.float(), gathered_targets.float().T) / self.dfc_temperature
        labels = torch.arange(anchors.shape[0], device=logits.device) + label_offset
        dfc_loss = F.cross_entropy(logits, labels)

        positive_logits = logits[torch.arange(anchors.shape[0], device=logits.device), labels]
        negative_mask = torch.ones_like(logits, dtype=torch.bool)
        negative_mask[torch.arange(anchors.shape[0], device=logits.device), labels] = False
        negative_logits = logits[negative_mask]
        contrastive_accuracy = (logits.argmax(dim=-1) == labels).float().mean()

        output = {
            "L_dfc": dfc_loss,
            "dfc_positive_logit": positive_logits.mean(),
            "dfc_negative_logit": negative_logits.mean() if negative_logits.numel() > 0 else logits.new_zeros(()),
            "dfc_accuracy": contrastive_accuracy,
            "sampled_delta_mean": gathered_deltas.mean(),
            "sampled_delta_min": gathered_deltas.min(),
            "sampled_delta_max": gathered_deltas.max(),
        }
        return output

    # Discretised state → instruction prefix (π₀.5 style); shared with QwenPI_v3.
    add_discretized_state_to_instruction = staticmethod(add_discretized_state_to_instruction)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model = Qwenvl_OFT(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"[train] Action Loss (with state): {action_loss.item()}")

    predict_output = model.predict_action(examples=[batch[0]])
    normalized_actions = predict_output["normalized_actions"]
    print(f"[infer] Predicted Action shape: {normalized_actions.shape}")

    # Backward-compat: examples without `state` should still work.
    sample_no_state = {k: v for k, v in sample.items() if k != "state"}
    forward_no_state = model([sample_no_state, sample_no_state])
    print(f"[train] Action Loss (no state): {forward_no_state['action_loss'].item()}")
    predict_no_state = model.predict_action(examples=[sample_no_state])
    print(f"[infer] Predicted Action shape (no state): {predict_no_state['normalized_actions'].shape}")

    print("Finished")
