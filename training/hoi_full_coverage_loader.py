"""Optional DDP-safe HOI loader that never drops a unique training sample.

The released comparison loader uses ``drop_last=True`` for structured DDP runs.
That was appropriate for controlled ablations, but it would discard part of the
final 5,462-window official corpus every epoch.

This module pads the shuffled global index list to a multiple of
``world_size * batch_size`` and then shards it across ranks. Consequently:

* every unique sample appears at least once per epoch;
* every rank executes the same number of full-size batches;
* trainer sample counters remain exact because no partial batch exists;
* only the minimal tail padding is repeated (at most one global batch minus one).

The default/released loader is unchanged unless ``--full-train-coverage`` is
explicitly enabled in the experimental entrypoint.
"""
from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Sized

import torch
from torch.utils.data import DataLoader, Sampler


def full_coverage_plan(total_samples: int, world_size: int, batch_size: int) -> dict[str, int]:
    """Return the exact padded epoch layout used by the final distributed run."""
    if total_samples < 1 or world_size < 1 or batch_size < 1:
        raise ValueError("total_samples, world_size, and batch_size must be positive")
    global_batch = int(world_size) * int(batch_size)
    batches_per_rank = int(math.ceil(total_samples / global_batch))
    sampler_total = batches_per_rank * global_batch
    samples_per_rank = batches_per_rank * int(batch_size)
    return {
        "unique_samples": int(total_samples),
        "global_batch_size": global_batch,
        "samples_per_rank": samples_per_rank,
        "sampler_total_samples": sampler_total,
        "padded_duplicate_samples": sampler_total - int(total_samples),
        "dropped_unique_samples": 0,
        "batches_per_rank": batches_per_rank,
        "last_batch_per_rank": int(batch_size),
    }


class FullBatchDistributedSampler(Sampler[int]):
    """Distributed sampler with full unique coverage and full per-rank batches."""

    def __init__(
        self,
        dataset: Sized,
        *,
        num_replicas: int,
        rank: int,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0,{num_replicas}), got {rank}")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.plan = full_coverage_plan(len(dataset), self.num_replicas, self.batch_size)

    def __iter__(self) -> Iterator[int]:
        size = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(size, generator=generator).tolist()
        else:
            indices = list(range(size))

        padding = self.plan["sampler_total_samples"] - size
        if padding > 0:
            # padding is always < one global batch and therefore < size for the
            # actual HOT3D corpus, but repeat defensively for synthetic tests.
            repeats = int(math.ceil(padding / max(size, 1)))
            indices.extend((indices * repeats)[:padding])
        if len(indices) != self.plan["sampler_total_samples"]:
            raise RuntimeError("full-coverage sampler produced the wrong global length")

        rank_indices = indices[self.rank : self.plan["sampler_total_samples"] : self.num_replicas]
        if len(rank_indices) != self.plan["samples_per_rank"]:
            raise RuntimeError("full-coverage sampler produced unequal rank lengths")
        return iter(rank_indices)

    def __len__(self) -> int:
        return int(self.plan["samples_per_rank"])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def install_full_coverage_hoi_loader() -> None:
    """Patch only this process's experimental HOI train-loader dispatch."""
    import dataset_components.dataloader as dataloader_module
    import dataset_components.hoi_dataset as hoi

    original = dataloader_module.build_hoi_dataloader

    def build_hoi_dataloader(args: Any, mode: str, rank: int = 0, world_size: int = 1):
        if mode != "train":
            return original(args, mode=mode, rank=rank, world_size=world_size)

        roots = args.hoi_train_dirs
        if not roots:
            return None, {}
        deterministic = bool(args.deterministic_train)
        dataset = hoi.HandObjectInteractionDataset(
            roots,
            hoi.HOIDatasetConfig(
                max_scene_points=args.max_scene_points,
                max_robot_points=args.max_robot_points,
                target_horizon=args.hoi_target_horizon,
                deterministic=deterministic,
                require_dino_features=args.hoi_require_dino_features,
                seed=args.seed + rank,
                domain=args.hoi_domain,
                mano_sidecar_root=getattr(args, "hoi_mano_sidecar_root", None),
            ),
        )

        sampler = FullBatchDistributedSampler(
            dataset,
            num_replicas=max(1, int(world_size)),
            rank=int(rank),
            batch_size=int(args.batch_size),
            shuffle=not deterministic,
            seed=int(args.seed),
        )
        plan = sampler.plan
        num_workers = 0 if deterministic else int(args.num_workers)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            # The sampler guarantees divisibility by batch_size, so this drops
            # nothing while protecting the trainer from an accidental partial.
            drop_last=True,
            collate_fn=partial(hoi.hoi_collate_fn, args=args),
        )
        info = {
            # The trainer defines epoch_count = processed_samples / total_samples.
            # Use the padded epoch size so exactly one loader pass equals 1.0 epoch.
            "total_samples": int(plan["sampler_total_samples"]),
            "unique_samples": int(len(dataset)),
            "total_batches_needed": int(plan["batches_per_rank"]),
            "batches_per_rank": int(plan["batches_per_rank"]),
            "world_size": int(world_size),
            "full_coverage": True,
            "padded_duplicate_samples": int(plan["padded_duplicate_samples"]),
            "dropped_unique_samples": 0,
        }
        if rank == 0:
            print(
                f"[hoi_npz:train/full-coverage] unique_samples={len(dataset)} "
                f"sampler_samples={plan['sampler_total_samples']} "
                f"batches_per_rank={plan['batches_per_rank']} "
                f"batch_per_rank={args.batch_size} "
                f"padded_duplicates={plan['padded_duplicate_samples']} "
                f"dropped_unique=0 roots={[str(Path(root)) for root in roots]}"
            )
        return loader, info

    dataloader_module.build_hoi_dataloader = build_hoi_dataloader
