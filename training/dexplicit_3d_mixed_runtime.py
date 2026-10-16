"""Full-dexplicit_3d mixed-data runtime; unchanged prediction graph and native trainer.

Full evaluations use native eval cadence. Quick panels run at intervening
resume-save boundaries and never enter native best-checkpoint selection.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from training.dexplicit_3d_mixed_manifest import DATASETS, METRIC, digest, write_json


@contextmanager
def preserve_rng():
    py, npstate = random.getstate(), np.random.get_state()
    with torch.random.fork_rng():
        try: yield
        finally:
            random.setstate(py); np.random.set_state(npstate)


def is_full(step, schedule):
    return step == 0 or step % schedule['full_every_steps'] == 0


def raw_metrics(records):
    """Point-error sums/counts, not a mean of window or dataset means."""
    result = {}
    for dataset in (*DATASETS, 'pooled'):
        rows = records if dataset == 'pooled' else [r for r in records if r['dataset'] == dataset]
        if not rows: raise ValueError(f'Evaluation omitted dataset {dataset}')
        total = sum(float(r[METRIC+'/sum']) for r in rows)
        count = sum(float(r[METRIC+'/count']) for r in rows)
        if (not math.isfinite(count) or count < 0 or count != int(count)
                or not math.isfinite(total) or total < 0 or (count == 0 and total != 0)):
            raise ValueError('Invalid metric numerator/denominator')
        result[dataset] = dict(sum=total, count=count, value=total/count if count else None,
            windows=len(rows), sources=len({(r['dataset'],r['source_clip_id']) for r in rows}))
    return result


def prune_resume_checkpoints(run, keep):
    """Bound rolling resume files after a durable save; retain all eval files.

    Aliases may be symlinks, hardlinks, or copies. Protect their target/inode and
    keep the newest rolling files; best checkpoint aliases remain readable.
    """
    if type(keep) is not int or keep < 1:
        raise ValueError('positive resume checkpoint retention required')
    run = Path(run)
    files = sorted((run/'checkpoints').glob('*.pt'), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    aliases = [p for p in run.glob('*.pt') if p.is_file()]
    protected = {p.resolve() for p in aliases}
    for path in files[keep:]:
        if path.resolve() in protected or any(path.samefile(alias) for alias in aliases):
            continue
        path.unlink()


def install_dataset(plan, original):
    def install(options):
        original(options)
        import dataset_components.hoi_dataset as hoi
        parent = hoi.HandObjectInteractionDataset
        records = {r['path']: r for r in plan['records']}
        class MixedDataset(parent):
            def __init__(self, roots, config):
                super().__init__(roots, config)
                for p in self.samples:
                    if str(Path(p).resolve()) not in records:
                        raise ValueError(f'Unplanned dataset file: {p}')
                roles = {records[str(Path(p).resolve())]['role'] for p in self.samples}
                if len(roles) != 1:
                    raise ValueError('A loader must not mix training and held-out samples')
                if roles == {'train'}:
                    # Fixed interleaving preserves the inherited exact-resume
                    # requirement sampler.shuffle=False and all-window coverage.
                    def ordering(path):
                        row = records[str(Path(path).resolve())]
                        return hashlib.sha256(f"2026:{row['dataset']}:{row['window']}".encode()).hexdigest()
                    self.samples.sort(key=ordering)
            def _assemble_sample(self, path, arrays, *args, **kwargs):
                sample = super()._assemble_sample(path, arrays, *args, **kwargs)
                row = records[str(Path(path).resolve())]
                sample['__dataset__'] = row['dataset']
                sample['source_clip_id'] = row['source']
                h = sample['scene_flows'].shape[0]
                sample.setdefault('action_is_unlabelled', torch.tensor(False))
                sample.setdefault('hand_joints_world', torch.zeros(h,2,21,3))
                sample.setdefault('hand_joint_valid', torch.zeros(h,2,21))
                has_mano = any(k in arrays for k in ('mano_pose_coeffs','mano_hand_pose','mano_thetas'))
                sample.setdefault('mano_valid', sample['hand_joint_valid'].bool().any(-1).float() if has_mano else torch.zeros(h,2))
                if row['dataset'] == 'phystwin':
                    if not bool(sample['action_is_unlabelled']):
                        raise ValueError('PhysTwin requires its tracked-control adapter')
                    sample['hand_joint_valid'] = torch.zeros_like(sample['hand_joint_valid'])
                    sample['mano_valid'] = torch.zeros_like(sample['mano_valid'])
                    if 'scene_delta_valid' not in sample:
                        raise ValueError('PhysTwin node compatibility/motion masks absent')
                return sample
        hoi.HandObjectInteractionDataset = MixedDataset
    return install


def install_training_hooks(trainer_module, plan):
    cls = trainer_module.Trainer
    if getattr(cls, '_mixed50_installed', False): raise RuntimeError('Install mixed hooks once')
    cls._mixed50_installed = True
    initial_setup, original_eval, original_save = cls.setup_dataloader, cls.eval_step, cls._save_latest_checkpoint_now
    rows = {r['path']: r for r in plan['records']}
    quick = set(map(tuple, plan['quick_windows']))
    plan_hash = digest(Path(plan['root'])/'plan.json')

    def setup(self):
        info = initial_setup(self)
        if self.world_size != 1: raise ValueError('This campaign uses one GPU/process')
        if len(self.train_dataloader) != plan['schedule']['steps_per_epoch']:
            raise ValueError('Actual loader size differs from 50-epoch schedule')
        sampler = self.train_dataloader.sampler
        if not hasattr(sampler, 'set_epoch') or getattr(sampler, 'shuffle', True):
            raise ValueError('Expected inherited deterministic full-coverage sampler')
        samples = self.test_dataloader.dataset.samples
        indices = [i for i,p in enumerate(samples) if (rows[str(Path(p).resolve())]['dataset'],
                                                       rows[str(Path(p).resolve())]['window']) in quick]
        if len(indices) != len(quick): raise ValueError('Quick panel identity is incomplete')
        # Subset of the SAME full dataset: sample/point seeds do not change.
        self._mixed_quick_loader = DataLoader(Subset(self.test_dataloader.dataset, indices),
            batch_size=self.args.batch_size, shuffle=False, drop_last=False,
            num_workers=self.args.eval_num_workers, collate_fn=self.test_dataloader.collate_fn,
            pin_memory=True, generator=torch.Generator().manual_seed(2026))
        self._mixed_full_loader = self.test_dataloader
        self._mixed_panel = False
        self._mixed_quick_running = False
        return info

    def evaluate(self, dataloader, data_iter, prefix, log_dict=None):
        if prefix != 'test': return original_eval(self, dataloader, data_iter, prefix, log_dict)
        panel = bool(self._mixed_panel)
        loader = self._mixed_quick_loader if panel else self._mixed_full_loader
        old_batches = self.args.num_eval_batches
        self.args.num_eval_batches = len(loader)
        try:
            with preserve_rng():
                result, iterator = original_eval(self, loader, iter(loader), prefix, dict(log_dict or {}))
        finally:
            self.args.num_eval_batches = old_batches
        pending = getattr(self, '_score_se3_pending_bundle', None)
        if not isinstance(pending, dict): raise RuntimeError('Native evaluator did not export per-window records')
        records = pending['records']
        expected = quick if panel else {(r['dataset'],r['window']) for r in plan['records'] if r['role']=='holdout'}
        actual = {(str(r['dataset']),str(r['sample_key'])) for r in records}
        if actual != expected or len(records) != len(expected):
            raise ValueError('Evaluation omitted/duplicated windows or mixed identities')
        source_lookup = {(r['dataset'],r['window']):r['source'] for r in plan['records']}
        if any(str(r['source_clip_id']) != source_lookup[(str(r['dataset']),str(r['sample_key']))] for r in records):
            raise ValueError('Evaluation source identity changed')
        values = raw_metrics(records)
        step = int(self.score_se3_campaign_step)
        kind = 'quick' if panel else 'full'
        prefix_out = 'quick' if panel else 'test'
        if panel:
            result = {(('quick/'+k[5:]) if k.startswith('test/') else k):v for k,v in result.items()}
            del self._score_se3_pending_bundle
        for name, value in values.items():
            stem = f'{prefix_out}/{name}/{METRIC}'
            if value['value'] is not None: result[stem] = value['value']
            result[stem+'/sum'], result[stem+'/count'] = value['sum'], value['count']
        result['mixed/epoch_exact'] = step / plan['schedule']['steps_per_epoch']
        result['mixed/full_evaluation'] = int(not panel)
        payload = dict(kind=kind, campaign_step=step, epoch=result['mixed/epoch_exact'],
                       plan_sha256=plan_hash, metrics=values)
        out = Path(self.save_dir)/'mixed_evaluations'
        # Replaceable monitoring only. Native full bundles remain immutable.
        write_json(out/f'{kind}_{step:08d}.json', payload, immutable=False)
        write_json(Path(self.save_dir)/'latest_test_metrics.json',
                   {'evaluation_kind':kind, 'metrics':result}, immutable=False)
        if not panel and step == 0 and not getattr(self.args, 'score_se3_eval_only', False):
            # An inferior epoch5 model must not displace the warm initializer.
            self._save_checkpoint_now(0, log_dict=result)
        if self.rank == 0:
            import wandb
            wandb.log(result)
            print(f'[mixed50] {kind} epoch={payload["epoch"]:.1f} raw-L2={values["pooled"]["value"]}', flush=True)
        return result, iterator

    def save_latest(self, log_dict=None):
        step = int(self.score_se3_campaign_step)
        due = step > 0 and step % plan['schedule']['steps_per_epoch'] == 0 and not is_full(step, plan['schedule'])
        logs = dict(log_dict or {})
        if due and not self._mixed_quick_running:
            self._mixed_quick_running = True; self._mixed_panel = True
            try:
                logs, _ = self.eval_step(self._mixed_quick_loader, None, 'test', logs)
            finally:
                self._mixed_panel = False; self._mixed_quick_running = False
        result = original_save(self, log_dict=logs)
        keep = plan.get('random30',{}).get('continuation',{}).get('resume_checkpoints_to_keep')
        if keep and self.rank == 0:
            prune_resume_checkpoints(self.save_dir, keep)
        return result

    cls.setup_dataloader, cls.eval_step, cls._save_latest_checkpoint_now = setup, evaluate, save_latest


def install(plan):
    """After carrier installation, before native trainer construction."""
    import train_hot3d_dynamics as entry
    import training.deformation_graph_model as factory
    import training.score_se3_integration as loading
    import training.score_se3_contract as contracts
    from training.dexplicit_3d_scaleup import build_raw_contract
    parent = factory.DeformationGraphExperimentalModel
    plan_hash = digest(Path(plan['root'])/'plan.json')
    class MixedDeXplicit3D(parent):
        def __init__(self, args, *a, **kw):
            args.dexplicit_3d_mixed50_plan_sha256 = plan_hash
            if getattr(args,'og_args',None) is not None: args.og_args.dexplicit_3d_mixed50_plan_sha256 = plan_hash
            super().__init__(args, *a, **kw)
    factory.DeformationGraphExperimentalModel = MixedDeXplicit3D
    entry.install_dataset_patch = install_dataset(plan, entry.install_dataset_patch)
    loading.SCORE_SE3_RESUME_CAUSAL_ARGS = tuple(dict.fromkeys((*loading.SCORE_SE3_RESUME_CAUSAL_ARGS, 'dexplicit_3d_mixed50_plan_sha256')))
    original_contract = contracts.build_score_se3_contract
    def build(args, **kwargs):
        result = build_raw_contract(original_contract, args, **kwargs)
        result['dexplicit_3d_mixed50'] = dict(plan_sha256=plan_hash, schedule=plan['schedule'],
            holdout_role='reused monitoring/validation holdout, not untouched final test',
            checkpoint_selection='full three-dataset raw moved L2 only', model='unchanged full bidirectional dexplicit_3d')
        return result
    contracts.build_score_se3_contract = build
    original_last_hook = entry._install_test_only_eval
    def final_hook(module):
        original_last_hook(module)
        install_training_hooks(module, plan)
    entry._install_test_only_eval = final_hook
