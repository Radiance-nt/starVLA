# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin, skip_first_batches
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def build_accelerator(cfg) -> Accelerator:
    grad_accum = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1))
    deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=grad_accum)
    grad_accum_plugin = GradientAccumulationPlugin(
        num_steps=grad_accum,
        sync_each_batch=True,
    )
    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_plugin=grad_accum_plugin,
    )
    accelerator.print(accelerator.state)
    return accelerator


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader, self.lr_scheduler = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.lr_scheduler,
        )

        if getattr(self, "resume_state_checkpoint", None):
            self._load_checkpoint(self.resume_state_checkpoint)
        else:
            self._adjust_lr_scheduler_for_resume()

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            init_timeout = int(os.environ.get("WANDB_INIT_TIMEOUT", "300"))
            wandb_run_id = os.environ.get("WANDB_RUN_ID") or None
            wandb_resume = os.environ.get("WANDB_RESUME") or None
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
                id=wandb_run_id,
                resume=wandb_resume,
                settings=wandb.Settings(init_timeout=init_timeout),
            )

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        self.resume_state_checkpoint = None

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                if os.path.isdir(self.resume_from_checkpoint):
                    self.resume_state_checkpoint = self.resume_from_checkpoint
                    logger.info(
                        f"Resuming full training state from checkpoint: {self.resume_state_checkpoint}, "
                        f"steps: {self.completed_steps}"
                    )
                else:
                    self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                    logger.warning(
                        "Resuming from a legacy model-only checkpoint without optimizer/scheduler state: "
                        f"{self.resume_from_checkpoint}, steps: {self.completed_steps}"
                    )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0 and not getattr(self, "resume_state_checkpoint", None):
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            self._fast_forward_lr_scheduler(self.completed_steps)
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _get_manual_lr_scheduler(self):
        """
        Return the underlying raw scheduler for StarVLA's historical step semantics.

        Accelerate wraps schedulers with `AcceleratedScheduler`, which can step the
        underlying scheduler multiple times per optimizer update when
        `split_batches=False`. Existing StarVLA runs were produced by stepping the
        raw scheduler once per optimizer step, so resume compatibility needs to
        preserve that behavior.
        """
        return getattr(self.lr_scheduler, "scheduler", self.lr_scheduler)

    def _fast_forward_lr_scheduler(self, steps: int):
        scheduler = self._get_manual_lr_scheduler()
        for _ in range(steps):
            scheduler.step()

    def _step_lr_scheduler(self):
        self._get_manual_lr_scheduler().step()

    def _ensure_legacy_scheduler_state(self, checkpoint_path: str):
        """
        Backfill `scheduler.bin` for old full-state checkpoints that were saved
        before StarVLA started serializing scheduler state.
        """
        scheduler_file = Path(checkpoint_path) / "scheduler.bin"
        if scheduler_file.exists():
            return

        if self.accelerator.is_main_process:
            logger.warning(
                "Legacy checkpoint is missing scheduler.bin; reconstructing scheduler state "
                f"from completed_steps={self.completed_steps} at {scheduler_file}"
            )
            self._fast_forward_lr_scheduler(self.completed_steps)
            torch.save(self._get_manual_lr_scheduler().state_dict(), scheduler_file)
            logger.info(f"Reconstructed scheduler state saved to {scheduler_file}")

        self.accelerator.wait_for_everyone()

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self._ensure_legacy_scheduler_state(checkpoint_path)
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        state_checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}_state")
        self.accelerator.save_state(state_checkpoint_path)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(
                f"✅ Checkpoint saved at {checkpoint_path} (full state: {state_checkpoint_path})"
            )

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

            self._prune_old_checkpoints()

        self.accelerator.wait_for_everyone()

    def _get_checkpoint_retention_limit(self):
        limit = getattr(self.config.trainer, "save_total_limit", None)
        if limit in (None, "", False):
            return None
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            logger.warning(f"Ignoring invalid trainer.save_total_limit={limit!r}")
            return None
        return limit if limit > 0 else None

    def _prune_old_checkpoints(self):
        limit = self._get_checkpoint_retention_limit()
        if limit is None:
            return

        checkpoint_entries = {}
        for entry in os.listdir(self.checkpoint_dir):
            entry_path = os.path.join(self.checkpoint_dir, entry)

            state_match = re.match(r"steps_(\d+)_state$", entry)
            if state_match and os.path.isdir(entry_path):
                checkpoint_entries.setdefault(int(state_match.group(1)), []).append(entry_path)
                continue

            weight_match = re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", entry)
            if weight_match and os.path.isfile(entry_path):
                checkpoint_entries.setdefault(int(weight_match.group(1)), []).append(entry_path)

        if len(checkpoint_entries) <= limit:
            return

        steps_to_prune = sorted(checkpoint_entries)[:-limit]
        for step in steps_to_prune:
            for entry_path in checkpoint_entries[step]:
                try:
                    if os.path.isdir(entry_path):
                        shutil.rmtree(entry_path)
                    else:
                        os.remove(entry_path)
                    logger.info(f"Pruned old checkpoint artifact: {entry_path}")
                except FileNotFoundError:
                    continue

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        if self.completed_steps > 0:
            self.vla_iter = self._build_resume_aligned_iterator(self.vla_train_dataloader, "vla_epoch_count")
        else:
            self.vla_epoch_count = 0
            self.vla_iter = iter(self.vla_train_dataloader)

    def _build_resume_aligned_iterator(self, dataloader, epoch_attr: str):
        """
        Reconstruct the dataloader position from completed optimizer steps.

        Accelerate/DeepSpeed restore model, optimizer, RNG, and scheduler state, but
        StarVLA does not checkpoint the dataloader iterator position. Without
        rebuilding the same epoch and intra-epoch batch offset, resumes from older
        checkpoints replay a different data order immediately.
        """
        try:
            steps_per_epoch = len(dataloader)
        except TypeError:
            self.accelerator.print(
                "Dataloader does not expose a stable length; falling back to iterator start on resume."
            )
            setattr(self, epoch_attr, 0)
            return iter(dataloader)

        if steps_per_epoch <= 0:
            setattr(self, epoch_attr, 0)
            return iter(dataloader)

        consumed_batches = self.completed_steps * self.accelerator.gradient_accumulation_steps
        resume_epoch = consumed_batches // steps_per_epoch
        resume_batch_offset = consumed_batches % steps_per_epoch

        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(resume_epoch)

        setattr(self, epoch_attr, resume_epoch)
        self.accelerator.print(
            "Restoring dataloader position from completed steps: "
            f"completed_steps={self.completed_steps}, "
            f"grad_accum={self.accelerator.gradient_accumulation_steps}, "
            f"resume_epoch={resume_epoch}, resume_batch_offset={resume_batch_offset}"
        )

        if resume_batch_offset > 0:
            return iter(skip_first_batches(dataloader, num_batches=resume_batch_offset))
        return iter(dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.accelerator.sync_gradients:
                if self.completed_steps % self.config.trainer.eval_interval == 0:
                    step_metrics = self.eval_action_model(step_metrics)

                step_metrics["data_time"] = t_end_data - t_start_data
                step_metrics["model_time"] = t_end_model - t_start_model
                self._log_metrics(step_metrics)

                if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                    self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(
                    batch_vla,
                    current_step=self.completed_steps,
                    forward_mode="action",
                )
                action_loss = output_dict["action_loss"]

            self.accelerator.backward(action_loss)

            total_loss_metric = output_dict.get("total_loss", action_loss).detach()
            base_model = self.accelerator.unwrap_model(self.model)
            run_mgv_this_step = base_model.mgv_enabled and base_model.mgv_module.should_run_mgv(self.completed_steps)
            if run_mgv_this_step:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    mgv_output = self.model.forward(
                        batch_vla,
                        current_step=self.completed_steps,
                        forward_mode="mgv",
                    )
                    mgv_loss_local = mgv_output.pop("_loss_mgv_local")
                self.accelerator.backward(base_model.mgv_module.lambda_mgv * mgv_loss_local)
                total_loss_metric = total_loss_metric + base_model.mgv_module.lambda_mgv * mgv_output["loss_mgv"].detach()
                output_dict.update(mgv_output)

            output_dict["total_loss"] = total_loss_metric

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            if self.accelerator.sync_gradients:
                self._step_lr_scheduler()

        metrics = {}
        for key, value in output_dict.items():
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                metrics[key] = value.item()
        metrics.setdefault("action_dit_loss", action_loss.item())
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    accelerator = build_accelerator(cfg)
    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
