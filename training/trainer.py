# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import subprocess
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime, timedelta
import wandb
from dataset_components.dataloader import build_dataloader
from tqdm.auto import tqdm
from utils import *
from torch.amp import GradScaler
from arguments import parse_args
from collections import defaultdict
from utils import _print, check_model_parameters_for_nan, check_tensor_for_nan, handle_nan_grad_norm
import random
import re
import sys
import traceback
import numpy as np
import math
from pointworld.base import BaseModel
from pointworld.checkpoint_contract import apply_model_contract_to_args, read_checkpoint_contract
from training import checkpointing as checkpointing_utils

class Trainer:
    def __init__(self, args, inference_only=False, data_info_dict=None):
        self.args = args
        self.inference_only = inference_only
        # Evaluation-only campaign runs still need the held-out dataloader, but
        # must not construct or restore an optimizer.  This is deliberately
        # separate from the legacy inference_only mode, which skips dataloaders.
        self.evaluation_only = bool(
            getattr(args, "score_se3_eval_only", False)
        )
        self.device = None
        self.world_size = 1
        self.rank = 0
        self.setup_distributed()
        # Successful optimizer updates and consumed score-SE3 campaign steps
        # are distinct, restart-safe counters.  continued_baseline/metric_loss Phase A advances the
        # campaign counter during its matched no-op without claiming an AdamW
        # update; cadence, phase transitions, and max_train_steps use campaign
        # steps.
        self.optimizer_step = 0
        self.score_se3_campaign_step = 0
        self.optimizer_phase = "single_phase"
        if bool(getattr(self.args, "score_se3_eval", False)):
            from training.score_se3_integration import (
                install_score_se3_checkpoint_loader,
            )

            install_score_se3_checkpoint_loader()

        # ----------------------------------------------------------------------------------
        # Setup save dir + wandb (simplified: train-from-scratch semantics)
        # ----------------------------------------------------------------------------------
        self.exp_name, self.wandb_id = self.setup_wandb(exp_name=self.args.exp_name)
        self.save_dir = self.setup_save_dir(exp_name=self.exp_name)
        checkpoint = self.load_checkpoint_from_path(self.args.model_path)
        if checkpoint is not None:
            context = f"training checkpoint '{self.args.model_path}'"
            model_contract, _ = read_checkpoint_contract(checkpoint, context=context)
            changed = apply_model_contract_to_args(
                self.args,
                model_contract,
                context=context,
                explicit_cli_dests=set(getattr(self.args, "_explicit_cli_dests", set())),
            )
            if changed and self.rank == 0:
                _print("Applied canonical checkpoint model_contract for model initialization.")

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32
        
        # Norm stats are restored from checkpoint on resume; no manifest handling needed.

        # ----------------------------------------------------------------------------------
        # Build dataloaders (with rank/world_size passed in) or infer dims from checkpoint
        # ----------------------------------------------------------------------------------
        if not self.inference_only:
            data_info_dict = self.setup_dataloader()
        else:
            # In inference-only mode, allow data_info_dict to be None and infer from checkpoint
            if data_info_dict is None:
                assert checkpoint is not None, (
                    "inference_only=True with no data_info_dict requires a valid checkpoint"
                )
                # Extract model state dict and infer projection input dims
                state = checkpoint.get('model')
                if state is None:
                    raise KeyError("Checkpoint missing model weights ('model')")
                scene_key = "scene_feature_encoder.scene_raw_feat_proj.weight"
                robot_key = "robot_proj.fc1.weight"
                if scene_key not in state:
                    raise KeyError(f"Checkpoint missing required key '{scene_key}'")
                if robot_key not in state:
                    raise KeyError(f"Checkpoint missing required key '{robot_key}'")
                scene_w = state[scene_key]
                robot_w = state[robot_key]
                data_info_dict = {
                    'scene_features_dim': int(scene_w.shape[1]),
                    'robot_features_dim': int(robot_w.shape[1]),
                }
        assert data_info_dict is not None, "data_info_dict must be present"

        # ----------------------------------------------------------------------------------
        # Build model and optimizer
        # ----------------------------------------------------------------------------------
        self.cpu_pg = None
        if self.args.distributed:
            self.cpu_pg = dist.new_group(backend="gloo")      # reuse everywhere
            self.eval_global_keys = None                           # populated once
        self.model = BaseModel(args, data_info_dict, rank=self.rank, cpu_pg=self.cpu_pg)
        self.model.to(self.device)
        # Establish the exact requires_grad partition and static optimizer
        # topology before DDP snapshots trainable parameters.  Optimizer
        # parameters remain the same objects after wrapping.
        if not self.inference_only and not self.evaluation_only:
            self._create_optimizer()
            # The scaler is causal optimizer state.  Construct it before
            # checkpoint loading so an exact score-SE3 resume can restore any
            # prior overflow/backoff history.
            self.grad_scaler = GradScaler(
                "cuda",
                enabled=torch.cuda.is_available(),
            )
        else:
            self.optimizer = None
            self.lr_scheduler = None
            self.score_se3_controller = None
            self.grad_scaler = None
        # Wrap in DDP if using multiple processes
        if self.args.distributed:
            action_mode = getattr(self.args, "action_conditioning_mode", None)
            find_unused_parameters = (
                getattr(self.args, "hand_condition_encoder", "none") == "joint_transformer"
                and getattr(self.args, "hand_condition_fusion", "append") == "replace"
            ) or action_mode in {"absolute_tokens", "delta_tokens"} or (
                getattr(self.args, "dynamics_experiment", "") == "structured"
            )
            self.model = DDP(
                self.model,
                device_ids=[self.device],
                output_device=self.device,
                find_unused_parameters=find_unused_parameters,
            )
        if self.rank == 0:
            wandb.watch(self.model, log=None)  # log='all' grad hook crashes on None grads from conditional branches
        # ----------------------------------------------------------------------------------
        # Load checkpoint if specified
        # ----------------------------------------------------------------------------------
        # initialize counters
        self.epoch_count = 0
        self.batch_count = 0
        self.curr_run_batch_count = 0
        self.sample_count = 0
        self.best_checkpoint_metric = None
        self.best_checkpoint_metric_name = None
        self.best_checkpoint_filename = None
        # load model state dict and update counters
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        if (
            bool(getattr(self.args, "score_se3_eval", False))
            and not self.inference_only
            and not self.evaluation_only
        ):
            from training.score_se3_data_resume import (
                restore_score_se3_rng_state,
                restore_score_se3_train_iterator,
            )

            restore_score_se3_train_iterator(self, checkpoint)
            # DataLoader iterator construction can consume Torch RNG.  Restore
            # the checkpointed per-rank streams only after the exact data
            # cursor has been reconstructed.
            restore_score_se3_rng_state(self, checkpoint)
        # Recompute epoch_count from samples seen if training
        if not self.inference_only and not self.evaluation_only:
            self.epoch_count = self.sample_count / float(self.train_total_samples)
        if bool(getattr(self.args, "score_se3_eval", False)):
            from training.score_se3_integration import (
                write_trainer_score_se3_contract,
            )

            write_trainer_score_se3_contract(self)

        # ----------------------------------------------------------------------------------
        # Misc setup
        # ----------------------------------------------------------------------------------
        if self.rank == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            log_dict = {
                "model/total_params": total_params,
            }
            for k, v in data_info_dict.items():
                log_dict[f'model/{k}'] = v
            for k, v in log_dict.items():
                _print(f'{k:<30}: {v}')
            _print(f'{"Experiment Name":<30}: {self.exp_name}')
            _print(f'{"Wandb ID":<30}: {self.wandb_id}')
            wandb.log(log_dict)
        
        # Initialize consecutive NaN grad norm counter
        self.consecutive_nan_grad_count = 0

    def setup_save_dir(self, exp_name=None):
        self.log_dir = self.args.log_dir
        if self.log_dir.startswith("s3://"):
            raise ValueError(f"S3 log_dir is not supported in release refactor: {self.log_dir}")
        self.save_dir = os.path.join(self.log_dir, exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        if self.rank == 0:
            _print(f'Experiment ID: {exp_name}')
            _print(f'Saving to: {self.save_dir}')
        return self.save_dir
    
    def setup_distributed(self):
        """Initialize Torch Distributed if requested and set device."""
        ### DDP CHANGE ###
        if self.args.distributed:
            timeout_min = int(os.environ.get("DIST_TIMEOUT_MIN", "30"))
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(minutes=timeout_min),
            )
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)
        else:
            self.device = torch.device(self.args.device)
            self.rank = 0
            self.world_size = 1
        if self.args.deterministic_train:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if getattr(self.args, "deterministic_algorithms", False):
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # set seed
        np.random.seed(self.args.seed + self.rank)
        torch.manual_seed(self.args.seed + self.rank)
        random.seed(self.args.seed + self.rank)
        torch.cuda.manual_seed(self.args.seed + self.rank)
        _print(f'Rank {self.rank + 1}/{self.world_size} using device {self.device}')

    def setup_wandb(self, exp_name=None):
        # Initialize wandb only on rank 0
        if self.rank == 0:
            # Only rank=0 does wandb.init
            disable_wandb = (not getattr(self.args, "wandb", False)) or os.environ.get('WANDB_MODE') == 'disabled'
            fallback_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            base_name = exp_name or getattr(self.args, "wandb_run_name", None) or fallback_name
            if bool(getattr(self.args, "unique_run_dir", False)) and not self.inference_only:
                run_id = getattr(self.args, "run_id", None) or datetime.now().strftime("%Y%m%d-%H%M%S")
                final_name = f"{base_name}__{run_id}"
            else:
                final_name = base_name
            # --wandb-run-name overrides the DISPLAY name only; exp_name (and
            # therefore save_dir) keeps the output-dir-derived identity so
            # artifact layouts never move when a run is renamed for wandb.
            wandb_display_name = (
                getattr(self.args, "wandb_run_name", None) or final_name
            )
            self.wandb_run = wandb.init(
                project=getattr(self.args, "wandb_project", "point-world"),
                name=wandb_display_name,
                mode=getattr(self.args, "wandb_mode", "online") if not disable_wandb else "disabled",
                config=self.args.og_args
            )
            self.exp_name = final_name or self.wandb_run.name or fallback_name
            self.wandb_id = self.wandb_run.id
        else:
            self.exp_name = None
            self.wandb_id = None

        # Broadcast exp_name from rank 0 to all other ranks
        if self.args.distributed:
            exp_name_list = [self.exp_name]
            dist.broadcast_object_list(exp_name_list, src=0)
            self.exp_name = exp_name_list[0]
            wandb_id_list = [self.wandb_id]
            dist.broadcast_object_list(wandb_id_list, src=0)
            self.wandb_id = wandb_id_list[0]
        
        return self.exp_name, self.wandb_id

    @staticmethod
    def _metric_token(value):
        token = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value).strip())
        return token.strip("_") or "unknown"

    def _accumulate_hoi_dataset_eval_metrics(self, metrics, counts, batch, outputs):
        if getattr(self.args, "dataset_type", None) != "hoi_npz":
            return
        dataset_labels = batch.get("__dataset__")
        if not dataset_labels:
            return

        pred = outputs["scene_flows"]
        gt = batch["gt_scene_flows"].to(pred.device)
        l2 = torch.norm(pred - gt, dim=-1)

        context = batch["scene_context_mask"].to(pred.device).squeeze(-1).bool()
        exists = batch["scene_exists"].to(pred.device).squeeze(-1).bool()
        supervised = batch["scene_supervised_mask"].to(pred.device).squeeze(-1).bool()
        moved = batch["scene_moved_mask"].to(pred.device).squeeze(-1).bool()
        static = batch["scene_static_mask"].to(pred.device).squeeze(-1).bool()
        valid = (~context) & exists & supervised

        for label in sorted(set(str(value) for value in dataset_labels)):
            batch_mask = torch.tensor(
                [str(value) == label for value in dataset_labels],
                dtype=torch.bool,
                device=pred.device,
            ).view(-1, 1, 1)
            prefix = f"dataset/{self._metric_token(label)}"
            for metric_name, mask in (
                ("l2/mean", valid),
                ("l2_moved/mean", valid & moved),
                ("l2_static/mean", valid & static),
            ):
                metric_mask = mask & batch_mask
                count = metric_mask.sum()
                if count > 0:
                    key = f"{prefix}/{metric_name}"
                    metrics[key] += l2[metric_mask].sum()
                    counts[key] += count
    
    def setup_dataloader(self):
        # Create train / test datasets with DDP slicing
        if self.rank == 0:
            _print('Setting up dataloaders...')
        
        start = time.time()
        train_override_splits = getattr(self.args, "train_splits", None)
        if self.rank == 0 and train_override_splits is not None:
            _print(f"Overriding training splits with: {train_override_splits}")
        self.train_dataloader, train_info = build_dataloader(
            args=self.args,
            mode='train',
            rank=self.rank,
            world_size=self.world_size,
            override_splits=train_override_splits,
        )
        if self.rank == 0: _print(f'Train dataset setup took {time.time() - start:.2f}s')
        
        start = time.time()
        self.test_dataloader, _ = build_dataloader(args=self.args, mode='test', rank=self.rank, world_size=self.world_size, force_resampled_eval=True)
        if self.test_dataloader is None:
            if self.args.eval_freq > 0:
                raise RuntimeError("Test dataloader missing but eval_freq > 0; cannot run evaluation.")
            self.test_iter = None
            if self.rank == 0:
                _print("Test dataset missing; test-time eval disabled.")
        else:
            self.test_iter = iter(self.test_dataloader)
            if self.rank == 0: _print(f'Test dataset setup took {time.time() - start:.2f}s')
        
        data_info_dict = {}
        # Sample a batch to inspect dimension info
        sample_iter = self.test_iter if self.test_iter is not None else iter(self.train_dataloader)
        batch = next(sample_iter)
        robot_features = batch['robot_features']  # (B, T, NR, DR)
        scene_features = batch['scene_features']  # (B, T, NS, DS)
        data_info_dict = {
            'robot_features_dim': robot_features.shape[-1],
            'scene_features_dim': scene_features.shape[-1],
            'data_keys': list(batch.keys())
        }

        # Store total samples for epoch estimation
        self.train_total_samples = train_info['total_samples']
        # The score-SE3 resume cursor validates the exact full-coverage layout,
        # not merely the aggregate sample count.
        self.train_dataloader_info = dict(train_info)

        # Keep cadence evaluation physically separate from the authoritative
        # training iterator.  Advancing ``train_iter`` during evaluation would
        # make the next optimizer batch depend on how many train-eval batches
        # happened to run, while the restart cursor intentionally describes
        # only committed campaign batches.
        self.train_iter = iter(self.train_dataloader)
        self.train_eval_iter = iter(self.train_dataloader)
        return data_info_dict

    def _save_checkpoint_now(self, adjusted_batch_count, log_dict=None, force=False):
        return checkpointing_utils.save_checkpoint_now(self, adjusted_batch_count, log_dict, force=force)

    def _save_latest_checkpoint_now(self, log_dict=None):
        return checkpointing_utils.save_latest_checkpoint_now(self, log_dict)

    def save_checkpoint(self, log_dict=None, force=False):
        return checkpointing_utils.save_checkpoint(self, log_dict, force=force)

    def load_checkpoint_from_path(self, model_path=None):
        return checkpointing_utils.load_checkpoint_from_path(self, model_path)

    def load_checkpoint(self, checkpoint):     
        return checkpointing_utils.load_checkpoint(self, checkpoint)

    @torch.no_grad()
    def eval_step(self, dataloader, data_iter, prefix, log_dict=None):
        """
        • dataloader : the DataLoader you want to sample from  
        • data_iter  : a *persistent* iterator you keep around  
        • prefix     : string put in front of every logged key  
        """
        self.model.eval()
        if log_dict is None:
            log_dict = {}

        device = self.device
        metrics = defaultdict(lambda: torch.tensor(0.0, device=device))
        counts  = defaultdict(lambda: torch.tensor(0,   device=device))
        last_batch = None
        last_outputs = None
        with tqdm(range(self.args.num_eval_batches),
                  desc=f'eval-{prefix}',
                  leave=False,
                  disable=self.rank != 0) as tbar:
            for _ in tbar:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Exhausted: recreate iterator and continue without interrupting training
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                         for k, v in batch.items()}

                with torch.autocast('cuda', dtype=self.amp_dtype):
                    outputs = self.model(batch, training=False)
                    _, loss_dict = (
                        self.model.module.loss_fn(outputs, batch, training=False)
                        if self.args.distributed else
                        self.model.loss_fn(outputs, batch, training=False)
                    )
                last_batch = batch
                last_outputs = outputs

                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1 and not torch.isnan(v):
                            metrics[k] += v * self.args.batch_size
                            counts[k]  += self.args.batch_size
                    elif isinstance(v, (int, float)) and not math.isnan(v):
                        metrics[k] += v * self.args.batch_size
                        counts[k]  += self.args.batch_size
                self._accumulate_hoi_dataset_eval_metrics(metrics, counts, batch, outputs)

        # all‑reduce
        if self.args.distributed:
            tmp = [None] * self.world_size
            dist.all_gather_object(tmp, list(metrics.keys()), group=self.cpu_pg)
            self.eval_global_keys = sorted(set().union(*tmp))

            for k in self.eval_global_keys:        # pad missing keys locally
                metrics.setdefault(k, torch.tensor(0.0, device=self.device))
                counts .setdefault(k, torch.tensor(0,   device=self.device))
                dist.all_reduce(metrics[k], group=self.cpu_pg)
                dist.all_reduce(counts[k], group=self.cpu_pg)

        # Write into log_dict with prefix
        for k in metrics.keys():
            if counts[k] > 0:
                log_dict[f'{prefix}/{k}'] = (metrics[k] / counts[k]).item()

        return log_dict, data_iter

    def should_do(self, action_freq, last_batch, curr_batch):
        """Helper function for cadence checks."""
        return action_freq > 0 and (curr_batch - last_batch >= action_freq)

    def _sync_skip_flag(self, skip):
        """If any rank wants to skip, all ranks skip — otherwise DDP allreduce hangs."""
        if not self.args.distributed:
            return bool(skip)
        flag = torch.tensor([1 if skip else 0], device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.cpu_pg)
        return bool(flag.item())

    def _dummy_ddp_backward(self, outputs, scaler):
        live = None
        for key in ("scene_flows", "pred", "log_var"):
            tensor = outputs.get(key) if isinstance(outputs, dict) else None
            if torch.is_tensor(tensor):
                live = tensor.sum() if live is None else live + tensor.sum()
        if live is None:
            live = torch.zeros((), device=self.device, requires_grad=True)
        scaler.scale(live * 0).backward()

    def _warmup_dino_cache(self):
        """One-time prebuild of the uncompressed DINO context-frame sidecars with a
        real progress bar, so the actual startup bottleneck is visible (and front-loaded)
        instead of hiding behind a train bar stuck at 0/N. Sharded across ranks and
        threaded per rank. Idempotent: re-runs are near-instant (existing sidecars skip)."""
        if getattr(self.args, "dataset_type", None) != "hoi_npz":
            return
        if not getattr(self.args, "dino_cache_warmup", True):
            return
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from dataset_components.hoi_dataset import build_dino_sidecar, HandObjectInteractionDataset

        horizon = int(self.args.hoi_target_horizon)
        seen, uniq = set(), []
        for dl in (self.train_dataloader, self.test_dataloader):
            ds = getattr(dl, "dataset", None) if dl is not None else None
            if isinstance(ds, HandObjectInteractionDataset):
                for p in ds.samples:
                    if p not in seen:
                        seen.add(p)
                        uniq.append(p)
        if not uniq:
            return

        shard = uniq[self.rank::self.world_size]  # disjoint slice per rank
        workers = max(1, int(getattr(self.args, "dino_cache_warmup_workers", 8)))
        if self.rank == 0:
            _print(f"[dino-cache] warming up {len(uniq)} clips across {self.world_size} rank(s), "
                   f"{workers} threads/rank (one-time; cached reads are ~16x faster)...")
        t0 = time.time()
        built = skipped = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(build_dino_sidecar, p, horizon) for p in shard]
            stream = as_completed(futures)
            if self.rank == 0:
                stream = tqdm(stream, total=len(shard), desc="Caching DINO sidecars",
                              unit="clip", leave=True)
            for fut in stream:
                try:
                    res = fut.result()
                    built += (res == "built")
                    skipped += (res == "skipped")
                except Exception:
                    pass  # never let cache warmup abort training
        if self.args.distributed:
            dist.barrier()
        if self.rank == 0:
            _print(f"[dino-cache] done in {time.time() - t0:.1f}s "
                   f"(rank0 shard: built={built}, skipped={skipped})")

    def train(self):

        if bool(getattr(self, "score_se3_eval_only_loaded", False)):
            raise RuntimeError(
                "score-SE3 checkpoint_role=eval is evaluation-only; training is forbidden"
            )

        self._warmup_dino_cache()

        scaler = self.grad_scaler
        if scaler is None:
            raise RuntimeError("training requires a constructed GradScaler")
        
        start_cadence_step = (
            int(self.score_se3_campaign_step)
            if bool(getattr(self.args, "score_se3_eval", False))
            else 0
        )
        score_run = bool(getattr(self.args, "score_se3_eval", False))
        checkpoint_role = str(
            getattr(self.args, "score_se3_checkpoint_role", "init")
        )
        if score_run:
            from training.score_se3_data_resume import score_se3_cadence_anchor
        if self.args.save_freq > 0:
            last_save_batch = (
                score_se3_cadence_anchor(
                    start_cadence_step,
                    self.args.save_freq,
                    checkpoint_role=checkpoint_role,
                    evaluate_initialization=False,
                )
                if score_run
                else (
                    0
                    if start_cadence_step == 0
                    else (start_cadence_step // self.args.save_freq)
                    * self.args.save_freq
                )
            )
        else:
            last_save_batch = -1
        # Step zero evaluates once.  Resumes keep the absolute campaign grid
        # (85, 170, ...) instead of shifting every later evaluation by the
        # arbitrary interruption step.
        if self.args.eval_freq > 0:
            last_eval_batch = (
                score_se3_cadence_anchor(
                    start_cadence_step,
                    self.args.eval_freq,
                    checkpoint_role=checkpoint_role,
                    evaluate_initialization=True,
                )
                if score_run
                else (
                    -self.args.eval_freq
                    if start_cadence_step == 0
                    else (start_cadence_step // self.args.eval_freq)
                    * self.args.eval_freq
                )
            )
        else:
            last_eval_batch = -1
        loss_dict = {'total_loss': 0.0}
        # Lightweight heartbeat so the run is visibly alive between tqdm updates / during eval.
        HEARTBEAT_SECS = 60
        last_heartbeat = time.time()
        last_hb_samples = 0
        first_batch_done = False
        # With an explicit step budget, max_train_steps is authoritative and
        # the loop stays alive for one boundary iteration so evaluation/save
        # at that exact step cannot be skipped at an epoch boundary.
        while self.args.max_train_steps > 0 or self.epoch_count < self.args.num_epochs:
            # Create progress bar for this epoch (rank 0 only)
            if self.rank == 0:
                tepoch = tqdm(total=len(self.train_dataloader),
                              desc=f'Initializing...',  # Will be updated in the loop
                              leave=False)
            
            # Iterate through one epoch using the shared train_iter
            for train_batch_idx in range(len(self.train_dataloader)):
                log_dict = dict()

                # Calculate adjusted batch count for frequency checks (equivalent to elapsed time)
                # Divide by world_size since each process contributes to the global batch count
                # This makes frequency checks work the same in both distributed and non-distributed settings
                if bool(getattr(self.args, "score_se3_eval", False)):
                    adjusted_batch_count = int(self.score_se3_campaign_step)
                else:
                    adjusted_batch_count = self.curr_run_batch_count // self.world_size

                # eval when batch count since last eval >= eval_freq
                if self.should_do(self.args.eval_freq, last_eval_batch, adjusted_batch_count):
                    if self.rank == 0:
                        _print(f"[eval] start @ batch {adjusted_batch_count} "
                               f"({self.args.num_eval_batches} batches x train_eval+test)...")
                    _eval_t0 = time.time()
                    if self.args.distributed:
                        dist.barrier()
                    # eval on train set
                    log_dict, self.train_eval_iter = self.eval_step(
                        self.train_dataloader,
                        self.train_eval_iter,
                        prefix='train_eval',
                        log_dict=log_dict
                    )
                    # eval on test set (if available)
                    if self.test_dataloader is not None:
                        log_dict, self.test_iter = self.eval_step(
                            self.test_dataloader,   self.test_iter,
                            prefix='test',
                            log_dict=log_dict
                        )
                    if self.args.distributed:
                        dist.barrier()
                    if (
                        bool(getattr(self.args, "score_se3_eval", False))
                        and self.test_dataloader is not None
                        and bool(
                            getattr(
                                self.args,
                                "score_se3_keep_eval_checkpoints",
                                False,
                            )
                        )
                    ):
                        retained = (
                            checkpointing_utils.save_score_se3_eval_checkpoint(
                                self, log_dict=log_dict
                            )
                        )
                        if self.rank == 0:
                            if not isinstance(retained, dict):
                                raise RuntimeError(
                                    "rank zero did not receive retained checkpoint metadata"
                                )
                            if bool(
                                getattr(
                                    self.args,
                                    "score_se3_export_per_window",
                                    False,
                                )
                            ):
                                from training.score_se3_eval import (
                                    write_pending_score_record_bundle,
                                )

                                write_pending_score_record_bundle(
                                    self,
                                    checkpoint_path=retained["path"],
                                    checkpoint_sha256=retained["sha256"],
                                )
                        elif hasattr(self, "_score_se3_pending_bundle"):
                            # Rank zero owns the durable export after gathering
                            # all records; other ranks must release their local
                            # pending copy before the next cadence.
                            del self._score_se3_pending_bundle
                    if self.rank == 0:
                        _ckpt_metric = log_dict.get(self.args.checkpoint_metric)
                        _msg = f"[eval] done in {time.time() - _eval_t0:.1f}s"
                        if _ckpt_metric is not None:
                            _msg += f" | {self.args.checkpoint_metric}={_ckpt_metric:.4e}"
                        _print(_msg)
                        last_heartbeat = time.time()
                    if (
                        getattr(self.args, "save_best_only", False)
                        and self.test_dataloader is not None
                        and adjusted_batch_count > 0
                    ):
                        self._save_checkpoint_now(adjusted_batch_count, log_dict=log_dict)
                    torch.cuda.empty_cache()
                    last_eval_batch = adjusted_batch_count

                # Periodic resume checkpoint (always written, independent of save_best_only).
                if self.should_do(self.args.save_freq, last_save_batch, adjusted_batch_count):
                    self._save_latest_checkpoint_now(log_dict=log_dict)
                    last_save_batch = adjusted_batch_count

                # Stop only after the boundary-step evaluation/checkpoint has
                # run.  Returning immediately after the update would skip the
                # scheduled step-850 validation and leave no final resume
                # checkpoint for the fixed campaign budget.
                if (
                    self.args.max_train_steps > 0
                    and adjusted_batch_count >= self.args.max_train_steps
                ):
                    if self.rank == 0:
                        tepoch.close()
                    return 0

                # Update progress bar with current time and stats (rank 0 only)
                if self.rank == 0:
                    global_batch_count = self.batch_count
                    global_sample_count = self.sample_count
                    current_time = datetime.now().strftime("%H:%M:%S")
                    tepoch.set_description(
                        f'[{current_time} {self.exp_name}], Epoch={self.epoch_count+1:.3f}/{self.args.num_epochs}, '
                        f'B={global_batch_count}, S={global_sample_count}, '
                        f'Loss={loss_dict["total_loss"]:.3e}'
                    )
                    tepoch.update(1)  # Update progress bar by 1

                    # Heartbeat: confirm the run is alive without flooding the terminal.
                    now = time.time()
                    if not first_batch_done:
                        first_batch_done = True
                        _print(f"[train] data flowing, first batch reached "
                               f"(B={global_batch_count}, S={global_sample_count}) "
                               f"\u2014 heartbeat every {HEARTBEAT_SECS}s")
                        last_heartbeat = now
                        last_hb_samples = global_sample_count
                    elif now - last_heartbeat >= HEARTBEAT_SECS:
                        dt = now - last_heartbeat
                        rate = (global_sample_count - last_hb_samples) / max(dt, 1e-9)
                        _print(f"[hb {current_time}] epoch={self.epoch_count+1:.2f}/{self.args.num_epochs} "
                               f"B={global_batch_count} S={global_sample_count} "
                               f"loss={loss_dict['total_loss']:.3e} {rate:.1f} samp/s")
                        last_heartbeat = now
                        last_hb_samples = global_sample_count

                # Cadence evaluation/checkpointing above describes the next
                # *unfetched* training batch.  Fetch only after the boundary
                # stop check, immediately before forward/backward, so an exact
                # resume never depends on an in-memory pending batch.
                if bool(getattr(self.args, "score_se3_eval", False)):
                    from training.score_se3_data_resume import (
                        ensure_score_se3_train_batch_fetch_allowed,
                    )

                    ensure_score_se3_train_batch_fetch_allowed(self)
                try:
                    train_batch = next(self.train_iter)
                except StopIteration:
                    sampler = getattr(self.train_dataloader, "sampler", None)
                    if hasattr(sampler, "set_epoch"):
                        if bool(getattr(self.args, "score_se3_eval", False)):
                            loader_length = int(len(self.train_dataloader))
                            next_epoch, offset = divmod(
                                int(self.score_se3_campaign_step),
                                loader_length,
                            )
                            if offset != 0:
                                raise RuntimeError(
                                    "score-SE3 train iterator exhausted away from an "
                                    "authoritative campaign epoch boundary: "
                                    f"campaign_step={self.score_se3_campaign_step}, "
                                    f"loader_length={loader_length}, offset={offset}"
                                )
                            sampler.set_epoch(next_epoch)
                        else:
                            sampler.set_epoch(int(self.epoch_count))
                    self.train_iter = iter(self.train_dataloader)
                    train_batch = next(self.train_iter)
                if bool(getattr(self.args, "score_se3_eval", False)):
                    from training.score_se3_data_resume import (
                        mark_score_se3_train_batch_fetched,
                    )

                    mark_score_se3_train_batch_fetched(self)
                self.model.train()
                train_batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in train_batch.items()}

                score_controller = getattr(self, "score_se3_controller", None)
                score_step_rates = None
                if score_controller is not None:
                    score_step_rates = score_controller.begin_batch(
                        self.model, self.optimizer
                    )
                else:
                    self.optimizer.zero_grad(set_to_none=True)

                try:
                    with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                        outputs = self.model(train_batch, training=True)

                        skip_nan_out = bool(outputs.get('_has_nan_outputs', False))

                    skip_nan_out = self._sync_skip_flag(skip_nan_out)
                    if skip_nan_out:
                        self._dummy_ddp_backward(outputs, scaler)
                        if score_run:
                            raise NaNDetectionError(
                                "NaN detected in synchronized score-SE3 model outputs",
                                context=(
                                    "training forward "
                                    f"(epoch {self.epoch_count:.3f}, "
                                    f"batch {train_batch_idx})"
                                ),
                            )
                        continue

                    with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                        total_loss, loss_dict = (
                            self.model.module.loss_fn(outputs, train_batch, training=True)
                            if self.args.distributed else self.model.loss_fn(outputs, train_batch, training=True)
                        )

                    # Detect before backward, then synchronize the decision so
                    # every DDP rank enters the same fail-closed handler.  A
                    # rank-local exception here would strand peers inside the
                    # subsequent gradient all-reduce.
                    loss_has_nan = check_tensor_for_nan(
                        total_loss,
                        "total_loss",
                        f"training step (epoch {self.epoch_count:.3f}, batch {train_batch_idx})",
                        raise_on_nan=False,
                    )
                    if self._sync_skip_flag(loss_has_nan):
                        raise NaNDetectionError(
                            "NaN detected in synchronized total_loss",
                            context=(
                                "training step "
                                f"(epoch {self.epoch_count:.3f}, batch {train_batch_idx})"
                            ),
                        )

                    scaler.scale(total_loss).backward()
                    optimizer_step_allowed = True
                    if score_controller is not None:
                        # Disabled groups must be cleared before unscale/step;
                        # otherwise AdamW creates moments and applies decoupled
                        # weight decay despite their zero learning rate.
                        optimizer_step_allowed = (
                            score_controller.prepare_optimizer_step(
                                self.optimizer
                            )
                        )

                    if optimizer_step_allowed:
                        scaler.unscale_(self.optimizer)
                        clip_parameters = (
                            score_controller.active_parameters(self.optimizer)
                            if score_controller is not None
                            else self.model.parameters()
                        )
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            clip_parameters,
                            max_norm=self.args.grad_clip_max_norm,
                        )
                    else:
                        # continued_baseline/metric_loss Phase A is an intentional matched no-op.  Do
                        # not call GradScaler.unscale_/step/update: unscale
                        # without update leaves scaler per-optimizer state live.
                        grad_norm = total_loss.detach().new_zeros(())

                    # Check gradient norm for NaN and skip batch if needed
                    should_skip_batch, self.consecutive_nan_grad_count = handle_nan_grad_norm(
                        grad_norm, self.consecutive_nan_grad_count, context=f"training step (epoch {self.epoch_count:.3f}, batch {train_batch_idx})"
                    )
                    should_skip_batch = self._sync_skip_flag(should_skip_batch)

                    if should_skip_batch and optimizer_step_allowed:
                        # Even when skipping, we must call scaler.step() and scaler.update()
                        # to reset the scaler state after unscale_() was called
                        scaler.step(self.optimizer)  # This will be a no-op due to NaN gradients
                        scaler.update()
                        if score_run:
                            raise NaNDetectionError(
                                "Non-finite synchronized score-SE3 gradient norm",
                                context=(
                                    "training backward "
                                    f"(epoch {self.epoch_count:.3f}, "
                                    f"batch {train_batch_idx})"
                                ),
                            )
                        continue  # Skip this batch and move to next iteration

                    # Check model parameters for NaN after gradient computation
                    check_model_parameters_for_nan(self.model, f"after backward pass (epoch {self.epoch_count:.3f}, batch {train_batch_idx})")

                    log_dict['grad_norm'] = float(grad_norm)
                    if score_controller is not None:
                        log_dict['optimizer/gradient_clipped'] = int(float(grad_norm) > float(self.args.grad_clip_max_norm))

                    optimizer_step_performed = False
                    if optimizer_step_allowed:
                        scale_before = float(scaler.get_scale())
                        scaler.step(self.optimizer)
                        scaler.update()
                        scale_after = float(scaler.get_scale())
                        if score_controller is not None:
                            from training.score_se3_finetune import (
                                optimizer_step_performed_after_grad_scaler,
                            )

                            optimizer_step_performed = (
                                not scaler.is_enabled()
                                or optimizer_step_performed_after_grad_scaler(
                                    scale_before_update=scale_before,
                                    scale_after_update=scale_after,
                                )
                            )
                        else:
                            optimizer_step_performed = True

                    if score_controller is not None:
                        log_dict['optimizer/update_performed'] = int(optimizer_step_performed)
                        score_controller.advance(
                            optimizer_step_performed=optimizer_step_performed,
                            optimizer=self.optimizer,
                        )
                        self.optimizer_step = int(
                            score_controller.optimizer_step
                        )
                        self.score_se3_campaign_step = int(
                            score_controller.campaign_step
                        )
                        self.optimizer_phase = (
                            score_controller.optimizer_phase
                        )
                        for group_name, rate in (score_step_rates or {}).items():
                            log_dict[f"lr/{group_name}"] = float(rate)
                    elif optimizer_step_performed:
                        self.optimizer_step += 1
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()
                            log_dict['lr'] = float(self.optimizer.param_groups[0]['lr'])

                    # Final check for NaN in model parameters after optimizer step
                    check_model_parameters_for_nan(self.model, f"after optimizer step (epoch {self.epoch_count:.3f}, batch {train_batch_idx})")
                
                except NaNDetectionError as nan_error:
                    # Handle NaN detection error
                    if self.rank == 0:
                        _print("\n" + "="*100)
                        _print("🚨 NaN DETECTED 🚨")
                        _print("="*100)
                        _print(f"Error: {nan_error}")
                        _print(f"Context: {nan_error.context}")
                        _print("="*100)
                    score_batch_pending = bool(
                        getattr(self.args, "score_se3_eval", False)
                    ) and bool(
                        getattr(self, "score_se3_data_batch_pending", False)
                    )
                    if score_batch_pending:
                        # A failure can occur after fetch, RNG consumption, or
                        # even an optimizer/scaler mutation but before the
                        # campaign counters commit.  Such a snapshot cannot
                        # truthfully identify its next training batch.  Leave
                        # the preceding cadence checkpoint untouched and fail
                        # closed on every rank instead of manufacturing an
                        # inexact emergency resume point.
                        if self.rank == 0:
                            _print(
                                "Exact score-SE3 emergency checkpoint skipped: "
                                "the current training batch is fetched but "
                                "uncommitted. Resume from the preceding cadence "
                                "checkpoint."
                            )
                    else:
                        # Every rank participates so distributed score-SE3 can
                        # gather one RNG snapshot per rank.  This path is used
                        # only when no score-SE3 batch is pending (or for legacy
                        # training where the exact cursor contract is absent).
                        self._save_latest_checkpoint_now(log_dict=log_dict)
                        if self.rank == 0:
                            _print(
                                "Emergency resume checkpoint saved to "
                                f"{self.save_dir}/model-latest.pt"
                            )
                    if self.rank == 0:
                        _print("="*100)
                        _print("Training stopped due to NaN detection.")
                        _print("Debug files saved to ./debug/ directory.")
                        _print("="*100)
                    
                    # Ensure all ranks are synchronized before exiting
                    if self.args.distributed:
                        dist.barrier()
                    
                    # Re-raise the error to stop training
                    raise nan_error

                # For aggregated metrics, you can all_reduce if desired, then average on rank=0.
                for k, v in loss_dict.items():
                    # convert NaN to 0 so reduce works; keep a flag if you still want NaN in logs
                    is_nan = isinstance(v, float) and math.isnan(v)
                    val_tensor = torch.tensor([0.0 if is_nan else float(v)],
                                            device=self.device, dtype=torch.float)

                    if self.args.distributed:
                        dist.all_reduce(val_tensor,
                                        op=dist.ReduceOp.SUM,
                                        group=self.cpu_pg)

                    avg_val = val_tensor.item() / self.world_size
                    log_dict[f"train/{k}"] = float('nan') if is_nan else avg_val
                # Increment counters
                # batch_count is the global count across all processes and runs
                # curr_run_batch_count is reset each time training starts
                # Both are multiplied by world_size to account for distributed training
                self.batch_count += 1 * self.world_size
                self.curr_run_batch_count += 1 * self.world_size
                if bool(getattr(self.args, "score_se3_eval", False)) and getattr(
                    self, "score_se3_controller", None
                ) is None:
                    self.score_se3_campaign_step += 1
                self.sample_count += self.args.batch_size * self.world_size
                if bool(getattr(self.args, "score_se3_eval", False)):
                    from training.score_se3_data_resume import (
                        commit_score_se3_train_batch,
                    )

                    commit_score_se3_train_batch(self)
                # Update epoch estimate continuously based on samples processed
                self.epoch_count = self.sample_count / float(self.train_total_samples)

                log_dict['batch_count'] = self.batch_count
                log_dict['epoch_count'] = self.epoch_count
                log_dict['curr_run_batch_count'] = self.curr_run_batch_count
                log_dict['sample_count'] = self.sample_count
                log_dict['optimizer_step'] = self.optimizer_step
                if bool(getattr(self.args, "score_se3_eval", False)):
                    log_dict['score_se3_campaign_step'] = self.score_se3_campaign_step

                # If we've reached target epochs mid-epoch, save final checkpoint and exit cleanly
                if (
                    self.args.max_train_steps <= 0
                    and self.epoch_count >= self.args.num_epochs
                ):
                    self._save_checkpoint_now(adjusted_batch_count, log_dict={})
                    if self.rank == 0:
                        tepoch.close()
                    return 0

                # Rank 0 logs to W&B
                if self.rank == 0:
                    wandb.log(log_dict)
                log_dict.clear()


            # Close progress bar for this epoch (rank 0 only)
            if self.rank == 0:
                tepoch.close()
            
        return 0

    @torch.no_grad()
    def full_eval(self, max_num_batches=1000):
        """
        Evaluate entire train/test sets, aggregated across all data.
        Metrics are computed across all GPUs and aggregated.
        """
        # barrier to ensure all processes have finished training
        if self.args.distributed:
            dist.barrier()
        
        # Switch to eval mode
        self.model.eval()

        # Helper to accumulate metrics across multiple batches and GPUs
        def accumulate_metrics(dataloader, prefix):
            metrics = defaultdict(lambda: torch.tensor(0.0, device=self.device))
            counts = defaultdict(lambda: torch.tensor(0, device=self.device))
            
            # Only show progress bar on rank 0
            iterator = dataloader
            if self.rank == 0:
                iterator = tqdm(dataloader, desc=f'Eval {prefix}', total=max_num_batches, leave=False)
            
            for i, batch in enumerate(iterator):
                if i >= max_num_batches:
                    break
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                # Forward pass
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                    outputs = self.model(batch, training=False)
                    total_loss, loss_dict = self.model.module.loss_fn(outputs, batch, training=False) if self.args.distributed else self.model.loss_fn(outputs, batch, training=False)
                
                # Accumulate batch metrics
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        if (v.numel() == 1 or v.dim() == 0) and not torch.isnan(v):
                            metrics[k] += v * self.args.batch_size
                            counts[k] += self.args.batch_size
                    elif isinstance(v, (int, float)) and not math.isnan(v):
                        metrics[k] += v * self.args.batch_size
                        counts[k] += self.args.batch_size

            # Aggregate metrics across GPUs if distributed
            if self.args.distributed:
                all_keys = [None] * self.world_size
                dist.all_gather_object(all_keys, list(metrics.keys()), group=self.cpu_pg)
                union_keys = set().union(*all_keys)
                for k in sorted(union_keys):
                    metrics.setdefault(k, torch.tensor(0.0, device=self.device))
                    counts.setdefault(k, torch.tensor(0,   device=self.device))
                    dist.all_reduce(metrics[k],
                                    op=dist.ReduceOp.SUM,
                                    group=self.cpu_pg)
                    dist.all_reduce(counts[k],
                                    op=dist.ReduceOp.SUM,
                                    group=self.cpu_pg)

            # Compute means
            aggregated_metrics = {}
            for k in metrics.keys():
                if counts[k] > 0:
                    aggregated_metrics[f"{prefix}/{k}"] = (metrics[k] / counts[k]).item()
                else:
                    aggregated_metrics[f"{prefix}/{k}"] = 0.0
            return aggregated_metrics

        # 1) Evaluate on train and test set
        train_metrics = accumulate_metrics(self.train_dataloader, prefix='full_eval/train')
        test_metrics = {}
        if self.test_dataloader is not None:
            test_metrics = accumulate_metrics(self.test_dataloader, prefix='full_eval/test')

        # Combine all metrics and log to W&B (only on rank 0)
        all_metrics = {**train_metrics, **test_metrics}
        if self.rank == 0:
            wandb.log(all_metrics)

        return all_metrics

    def _create_optimizer(self):
        """Create the declared optimizer before optional DDP wrapping."""

        model = self.model.module if isinstance(self.model, DDP) else self.model
        if bool(getattr(self.args, "score_se3_eval", False)):
            from training.score_se3_finetune import (
                ScoreSE3FineTuneConfig,
                prepare_score_se3_finetuning,
            )

            config = ScoreSE3FineTuneConfig(
                phase_a_steps=int(self.args.score_se3_phase_a_steps),
                total_steps=int(self.args.score_se3_total_steps),
                warmup_steps=int(self.args.score_se3_warmup_steps),
                phase_a_adapter_lr=float(self.args.score_se3_adapter_lr),
                phase_b_adapter_lr=float(
                    self.args.score_se3_adapter_phase_b_lr
                ),
                phase_b_base_lr=float(self.args.score_se3_base_lr),
                weight_decay=float(self.args.weight_decay),
                schedule_mode=str(getattr(self.args, "score_se3_schedule_mode", "staged_cosine")),
                hold_steps=int(getattr(self.args, "score_se3_hold_steps", 0)),
                min_lr_ratio=float(getattr(self.args, "score_se3_min_lr_ratio", 0.0)),
            )
            setup = prepare_score_se3_finetuning(
                model,
                str(self.args.score_se3_candidate),
                config=config,
                campaign_step=int(self.score_se3_campaign_step),
                optimizer_step=int(self.optimizer_step),
            )
            self.optimizer = setup.optimizer
            self.score_se3_controller = setup.controller
            # The controller is scheduler-like (state_dict/load_state_dict and
            # get_last_lr), but it owns phase safety as well as learning rates.
            self.lr_scheduler = setup.controller
            self.optimizer_phase = setup.controller.optimizer_phase
            self.score_se3_optimizer_plan = setup.controller.optimizer_plan()
            self.args.score_se3_optimizer_plan = self.score_se3_optimizer_plan
            self.args.score_se3_trainable_parameter_names = list(
                setup.parameter_plan.trainable_parameter_names
            )
            self.args.score_se3_initial_optimizer_phase = setup.controller.phase_at(0)
            self.args.score_se3_optimizer_class = (
                "torch.optim.adamw.AdamW"
            )
            self.args.score_se3_optimizer_groups = [
                {
                    "name": group.name,
                    "parameter_names": list(group.parameter_names),
                    "lr": float(self.optimizer.param_groups[index]["lr"]),
                    "weight_decay": float(
                        self.optimizer.param_groups[index]["weight_decay"]
                    ),
                }
                for index, group in enumerate(setup.parameter_plan.groups)
            ]
            if self.rank == 0:
                _print("\nScore-SE3 staged optimizer parameters:")
                for group in setup.parameter_plan.groups:
                    count = sum(
                        dict(model.named_parameters())[name].numel()
                        for name in group.parameter_names
                    )
                    _print(f"  {group.name}: {count:,} elements")
            return

        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.args.base_lr,
            weight_decay=self.args.weight_decay,
        )
        # Optional cosine decay over LR_SCHEDULE_TOTAL_STEPS optimizer steps
        # (env-controlled; default off = constant LR, the historical recipe).
        self.lr_scheduler = None
        self.score_se3_controller = None
        import os as _os
        if _os.environ.get("LR_SCHEDULE", "none") == "cosine":
            total = int(_os.environ.get(
                "LR_SCHEDULE_TOTAL_STEPS",
                str(max(1, int(getattr(self.args, "max_train_steps", 0) or 1))),
            ))
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total, eta_min=0.0
            )
            if self.rank == 0:
                _print(f"LR schedule: cosine over {total} steps from {self.args.base_lr}")
        if self.rank == 0:
            total_count = sum(p.numel() for p in params)
            _print("\nOptimizer parameters:")
            _print(f"  Total: {total_count:,} elements\n")
