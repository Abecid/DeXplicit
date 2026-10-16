"""Reselected epoch views on an immutable training pool, no motion filtering.

The native full-coverage cursor covers the selected *epoch view*, not the whole
multi-dataset pool. Its inventory fingerprint binds that pool plus this policy.
Training uses num_workers=0, as required by the existing exact-resume trainer.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import random
from torch.utils.data import Dataset
from training.hoi_full_coverage_loader import FullBatchDistributedSampler
from training.dexplicit_3d_scaleup import freeze_json

POLICY = 'uniform_per_dataset_without_replacement_per_epoch_v1'


def selected_indices(groups, epoch, *, count=1000, seed=2026):
    if type(epoch) is not int or epoch < 0 or type(count) is not int or count < 1:
        raise ValueError('nonnegative integer epoch and positive quota required')
    if set(groups) != {'hot3d', 'oakink2', 'phystwin'}:
        raise ValueError('all three dataset pools are required')
    all_ids = [i for group in groups.values() for i in group]
    if len(set(all_ids)) != len(all_ids) or not groups['phystwin']:
        raise ValueError('pools must be disjoint and PhysTwin nonempty')
    out = []
    for dataset in ('hot3d', 'oakink2', 'phystwin'):
        ids = sorted(groups[dataset])
        if dataset == 'phystwin':
            out.extend(ids)
            continue
        if len(ids) < count:
            raise ValueError(f'{dataset}: {len(ids)} windows < requested {count}')
        salt = hashlib.sha256(f'{POLICY}:{seed}:{epoch}:{dataset}'.encode()).digest()
        out.extend(random.Random(int.from_bytes(salt, 'big')).sample(ids, count))
    salt = hashlib.sha256(f'{POLICY}:{seed}:{epoch}:interleave'.encode()).digest()
    random.Random(int.from_bytes(salt, 'big')).shuffle(out)
    return out


class RandomEpochView(Dataset):
    """Finite, fixed-size epoch slots mapped to immutable native sample indices."""
    def __init__(self, base, records, root, *, count=1000, seed=2026, batch_size=16, epoch_offset=0):
        self.base = base
        self.root = Path(root)
        self.count, self.seed, self.batch_size = count, seed, batch_size
        if type(epoch_offset) is not int or epoch_offset < 0:
            raise ValueError('epoch_offset must be a nonnegative integer')
        self.epoch_offset = epoch_offset
        by_path = {str(Path(r['path']).resolve()): r for r in records if r['role'] == 'train'}
        self.pool = []
        self.groups = {d: [] for d in ('hot3d', 'oakink2', 'phystwin')}
        for i, p in enumerate(base.samples):
            path = str(Path(p).resolve())
            if path not in by_path:
                raise ValueError(f'nontraining or unplanned sample in random pool: {p}')
            row = by_path[path]
            self.pool.append(row)
            self.groups[row['dataset']].append(i)
        if len(self.pool) != len(by_path) or len({r['path'] for r in self.pool}) != len(self.pool):
            raise ValueError('native loader omitted or duplicated the frozen full training pool')
        # Native cursor supports an explicit immutable inventory override. Include
        # ALL candidates and policy; a changing epoch subset is not the identity.
        self.score_se3_inventory_records = [dict(policy=POLICY, seed=seed, quota=count,
            batch_size=batch_size, coverage_scope='selected_epoch_view', all_phystwin=True), *self.pool]
        if epoch_offset:
            self.score_se3_inventory_records[0]['epoch_offset'] = epoch_offset
        self.set_epoch(0)

    def set_epoch(self, epoch):
        ids = selected_indices(self.groups, epoch + self.epoch_offset, count=self.count, seed=self.seed)
        padding = (-len(ids)) % self.batch_size
        rows = [self.pool[i] for i in ids]
        receipt = dict(schema_version=1, policy=POLICY, zero_based_epoch=epoch,
            seed=self.seed, quota=self.count, motion_filter=False,
            windows=[dict(dataset=r['dataset'], source=r['source'], window=r['window'], path=r['path']) for r in rows],
            counts={d: sum(r['dataset'] == d for r in rows) for d in self.groups},
            padded_duplicate_paths=[r['path'] for r in rows[:padding]])
        if self.epoch_offset:
            receipt['sampling_zero_based_epoch'] = epoch + self.epoch_offset
        freeze_json(self.root / 'epoch_rosters' / f'epoch_{epoch+1:03d}.json', receipt)
        self.epoch, self.indices = epoch, ids

    @property
    def samples(self):
        return [self.base.samples[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.base[self.indices[index]]


class RandomEpochSampler(FullBatchDistributedSampler):
    def __init__(self, dataset, **kwargs):
        if kwargs.get('shuffle') or kwargs.get('num_replicas') != 1:
            raise ValueError('random30 uses one deterministic process and native resume cursor')
        super().__init__(dataset, **kwargs)
        self.plan.update(coverage_scope='selected_epoch_view', selection_policy=POLICY,
                         full_training_pool_size=len(dataset.base))

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.dataset.set_epoch(self.epoch)


def install(plan):
    """Add epoch views before native mixed loader construction, not after it."""
    import training.dexplicit_3d_mixed_runtime as mixed
    import training.hoi_full_coverage_loader as coverage
    original_install = mixed.install_dataset
    original_sampler = coverage.FullBatchDistributedSampler

    def dataset_installer(current_plan, original):
        base_install = original_install(current_plan, original)
        def setup(options):
            base_install(options)
            import dataset_components.hoi_dataset as hoi
            native_dataset = hoi.HandObjectInteractionDataset
            by_path = {r['path']: r for r in current_plan['records']}
            class construct(native_dataset):
                # Stays a subclass of the native dataset so trainer-side
                # isinstance() checks (e.g. the DINO cache warm-up) keep working;
                # __new__ returns the untouched native held-out dataset or the
                # per-epoch random training view.
                def __new__(cls, roots, config):
                    base = native_dataset(roots, config)
                    roles = {by_path[str(Path(p).resolve())]['role'] for p in base.samples}
                    if roles != {'train'}:
                        return base  # Held-out population and point sampling untouched.
                    policy = current_plan['random30']
                    return RandomEpochView(base, current_plan['records'], current_plan['root'],
                        count=policy['per_dataset_windows'], seed=policy['seed'],
                        batch_size=current_plan['schedule']['batch_size'],
                        epoch_offset=policy.get('epoch_offset', 0))
            hoi.HandObjectInteractionDataset = construct
        return setup

    def sampler(dataset, **kwargs):
        return (RandomEpochSampler if isinstance(dataset, RandomEpochView) else original_sampler)(dataset, **kwargs)

    mixed.install_dataset = dataset_installer
    coverage.FullBatchDistributedSampler = sampler
    mixed.install(plan)
