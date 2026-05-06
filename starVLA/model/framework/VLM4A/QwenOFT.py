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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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

    future_tokens: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "horizons": [],
            "order": "reverse",
            "use_future_tokens_for_action": True,
            "add_horizon_embedding": True,
        }
    )

    contrastive: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "loss_type": "infonce",
            "projector_hidden_dim": 2048,
            "projector_out_dim": 128,
            "temperature": 0.1,
            "loss_weight": 0.1,
            "future_encoder": "stopgrad_same_vlm",
            "negatives": "batch_only",
        }
    )


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

        future_cfg = self.config.framework.get("future_tokens", {})
        contrastive_cfg = self.config.framework.get("contrastive", {})
        self.future_tokens_enabled = bool(future_cfg.get("enabled", False))
        self.contrastive_enabled = bool(contrastive_cfg.get("enabled", False))
        self.future_loss_type = str(contrastive_cfg.get("loss_type", "infonce")).lower()
        self.future_loss_weight = float(contrastive_cfg.get("loss_weight", 0.1))
        self.future_temperature = float(contrastive_cfg.get("temperature", 0.1))
        self.use_future_tokens_for_action = bool(future_cfg.get("use_future_tokens_for_action", True))
        self.add_horizon_embedding = bool(future_cfg.get("add_horizon_embedding", True))
        self.future_token_order = str(future_cfg.get("order", "reverse")).lower()
        self.future_horizons = self._prepare_future_horizons(future_cfg.get("horizons", []))

        if self.contrastive_enabled and not self.future_tokens_enabled:
            raise ValueError("framework.contrastive.enabled=true requires framework.future_tokens.enabled=true")

        self.future_token_chars: list[str] = []
        self.future_token_ids: list[int] = []
        self.future_horizon_embedding = None
        self.future_condition_fuser = None
        self.future_projector = None

        if self.future_tokens_enabled:
            if len(self.future_horizons) > len(SAFE_FUTURE_TOKEN_CHARS):
                raise ValueError(
                    f"Configured {len(self.future_horizons)} future horizons, but only "
                    f"{len(SAFE_FUTURE_TOKEN_CHARS)} verified single-token markers are available."
                )

            self.future_token_chars = SAFE_FUTURE_TOKEN_CHARS[: len(self.future_horizons)]
            for token_char in self.future_token_chars:
                token_ids = self.qwen_vl_interface.processor.tokenizer(
                    token_char, add_special_tokens=False
                )["input_ids"]
                if len(token_ids) != 1:
                    raise ValueError(f"Future marker {token_char!r} must tokenize to exactly one token, got {token_ids}")
                self.future_token_ids.append(token_ids[0])

            if self.add_horizon_embedding:
                self.future_horizon_embedding = nn.Embedding(len(self.future_horizons), self.hidden_size)

            if self.use_future_tokens_for_action:
                self.future_condition_fuser = nn.Sequential(
                    nn.LayerNorm(self.hidden_size * 2),
                    nn.Linear(self.hidden_size * 2, self.hidden_size),
                    nn.GELU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                )

            if self.contrastive_enabled:
                projector_hidden_dim = int(contrastive_cfg.get("projector_hidden_dim", self.hidden_size))
                projector_out_dim = int(contrastive_cfg.get("projector_out_dim", 128))
                self.future_projector = nn.Sequential(
                    nn.LayerNorm(self.hidden_size),
                    nn.Linear(self.hidden_size, projector_hidden_dim),
                    nn.GELU(),
                    nn.Linear(projector_hidden_dim, projector_out_dim),
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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action token embeddings as action prediction queries
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]
            future_hidden = None
            if self.future_tokens_enabled:
                future_hidden = self._gather_specific_token_embeddings(
                    last_hidden, input_ids, self.future_token_ids
                )
                future_hidden = self._apply_horizon_embedding(future_hidden)
                action_queries = self._condition_action_queries(action_queries, future_hidden)

            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

            # Label alignment: take the last chunk_len segment
            actions = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            # Compute L1 loss
            action_loss = self.l1_loss(pred_actions, actions_target)

            total_loss = action_loss
            output_dict = {"action_loss": action_loss}

            if self.contrastive_enabled:
                future_loss = self._compute_future_auxiliary_loss(
                    examples=examples,
                    base_instructions=base_instructions,
                    future_hidden=future_hidden,
                )
                total_loss = total_loss + self.future_loss_weight * future_loss
                output_dict["future_loss"] = future_loss

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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action token embeddings as action prediction queries
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]
            if self.future_tokens_enabled:
                future_hidden = self._gather_specific_token_embeddings(
                    last_hidden, input_ids, self.future_token_ids
                )
                future_hidden = self._apply_horizon_embedding(future_hidden)
                action_queries = self._condition_action_queries(action_queries, future_hidden)
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

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

    def _prepare_future_horizons(self, horizons: List[int]) -> list[int]:
        horizons = [int(h) for h in horizons]
        if not horizons:
            return []
        if self.future_token_order == "reverse":
            return sorted(horizons, reverse=True)
        if self.future_token_order == "forward":
            return sorted(horizons)
        return horizons

    def _build_base_instructions(self, instructions: List[str], state: Optional[List[np.ndarray]]) -> List[str]:
        return self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions

    def _append_action_and_future_prompt(self, instructions: List[str]) -> List[str]:
        future_prompt = ""
        if self.future_tokens_enabled and self.future_token_chars:
            marker_text = "".join(f"[{token_char}]" for token_char in self.future_token_chars)
            future_prompt = f" Future context tokens (far-to-near): {marker_text}."

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

    def _apply_horizon_embedding(self, future_hidden: torch.Tensor) -> torch.Tensor:
        if self.future_horizon_embedding is None:
            return future_hidden

        horizon_index = torch.arange(len(self.future_horizons), device=future_hidden.device)
        horizon_emb = self.future_horizon_embedding(horizon_index).unsqueeze(0)
        return future_hidden + horizon_emb

    def _condition_action_queries(
        self,
        action_queries: torch.Tensor,
        future_hidden: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if future_hidden is None or self.future_condition_fuser is None:
            return action_queries

        future_context = future_hidden.mean(dim=1, keepdim=True).expand(-1, action_queries.shape[1], -1)
        fused_input = torch.cat([action_queries, future_context], dim=-1)
        return self.future_condition_fuser(fused_input)

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

    def _get_future_images_for_horizon(self, examples: List[dict], horizon: int) -> List[List[Image.Image]]:
        batch_images = []
        for example in examples:
            future_images = example.get("future_images", None)
            if future_images is None:
                raise ValueError(
                    "RC-CAFT future loss requires `future_images` in the batch. "
                    "Set datasets.vla_data.return_future_obs=true and future_obs_horizons accordingly."
                )

            if horizon in future_images:
                batch_images.append(future_images[horizon])
            elif str(horizon) in future_images:
                batch_images.append(future_images[str(horizon)])
            else:
                raise KeyError(f"Missing future horizon {horizon} in sample future_images keys={list(future_images.keys())}")
        return batch_images

    def _encode_future_targets(
        self,
        examples: List[dict],
        base_instructions: List[str],
    ) -> Dict[int, torch.Tensor]:
        if not self.future_horizons:
            return {}

        batch_size = len(examples)
        flat_future_images: List[List[Image.Image]] = []
        flat_instructions: List[str] = []

        for horizon in self.future_horizons:
            future_images = self._get_future_images_for_horizon(examples, horizon)
            flat_future_images.extend(future_images)
            flat_instructions.extend(base_instructions)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=flat_future_images,
            instructions=flat_instructions,
        )

        with torch.no_grad():
            future_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            pooled_future = self._pool_hidden_states(
                future_outputs.hidden_states[-1], qwen_inputs.get("attention_mask", None)
            )

        target_embeddings = {}
        for idx, horizon in enumerate(self.future_horizons):
            start = idx * batch_size
            end = start + batch_size
            target_embeddings[horizon] = pooled_future[start:end]

        return target_embeddings

    def _compute_future_auxiliary_loss(
        self,
        examples: List[dict],
        base_instructions: List[str],
        future_hidden: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if future_hidden is None or self.future_projector is None:
            raise ValueError("Future auxiliary loss requested but future token branch is not initialized")

        target_embeddings = self._encode_future_targets(examples, base_instructions)
        losses = []

        for idx, horizon in enumerate(self.future_horizons):
            anchor = self.future_projector(future_hidden[:, idx, :])
            target = self.future_projector(target_embeddings[horizon]).detach()

            anchor = F.normalize(anchor.float(), dim=-1)
            target = F.normalize(target.float(), dim=-1)

            if self.future_loss_type == "l2":
                loss_h = F.mse_loss(anchor, target)
            elif self.future_loss_type == "cosine":
                loss_h = 1.0 - F.cosine_similarity(anchor, target, dim=-1).mean()
            elif self.future_loss_type == "infonce":
                logits = torch.matmul(anchor, target.T) / self.future_temperature
                labels = torch.arange(logits.shape[0], device=logits.device)
                loss_h = F.cross_entropy(logits, labels)
            else:
                raise ValueError(f"Unsupported future loss type: {self.future_loss_type}")

            losses.append(loss_h)

        return torch.stack(losses).mean()

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
