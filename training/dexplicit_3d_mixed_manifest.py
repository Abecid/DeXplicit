"""Full-dexplicit_3d campaign inventory, split and schedule; no model changes."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import zipfile
import numpy as np

DATASETS = ('hot3d', 'oakink2', 'phystwin')
METRIC = 'l2_moved/mean_t01_t34'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, payload, *, immutable=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if immutable and path.exists():
        if path.read_text() != text: raise ValueError(f'Frozen receipt changed: {path}')
        return
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(text); os.replace(tmp, path)


def shape(path, name):
    with zipfile.ZipFile(path) as archive, archive.open(name + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        return reader(stream)[0]


def default_spec(main):
    p = Path(main).expanduser().resolve() / 'data/processed'
    hot = p / 'hot3d_clips_pointwise_dino'; oak = p / 'oakink2_35f_v2'; phy = p / 'phystwin/dynamics'
    return {'version': 1, 'seed': 2026, 'datasets': [
        {'dataset': 'hot3d', 'step_seconds': 1/15,
         'roots': [str(hot/'train_full_aria35f_v1'), str(hot/'metric_aligned_se3/validation'), str(hot/'metric_aligned_se3/final_test')],
         'holdout_roots': [str(hot/'metric_aligned_se3/validation'), str(hot/'metric_aligned_se3/final_test')],
         'protect_train_roots': [str(hot/'train_full_aria35f_v1')],
         'source_map_files': [str(hot/'train_source_map.json')], 'allow_hot3d_clip_filename': True},
        {'dataset': 'oakink2', 'step_seconds': 1/15,
         'roots': [str(oak/s) for s in ('train', 'validation', 'final_test')],
         'holdout_roots': [str(oak/'final_test')], 'source_map_files': [str(oak/'train_source_map.json')]},
        {'dataset': 'phystwin', 'step_seconds': 1/15,
         'roots': [str(phy/s) for s in ('train', 'val', 'test')], 'holdout_roots': [str(phy/'test')]}
    ]}


def split_sources(sources, pinned=(), protected=(), *, seed=2026):
    """Approximately 90:10 by independent sources; retain previous holdouts."""
    sources, pinned, protected = set(sources), set(pinned), set(protected)
    if len(sources) < 2 or not (pinned | protected) <= sources or pinned & protected:
        raise ValueError('Need >=2 sources and disjoint existing pinned/protected identities')
    target = max(1, int(len(sources) * .1 + .5))
    order = sorted(sources - pinned - protected, key=lambda s: hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest())
    holdout = pinned | set(order[:max(0, target-len(pinned))])
    if not holdout or holdout == sources: raise ValueError('No nonempty source-disjoint split')
    return sources-holdout, holdout


def _source(path, entry, metadata, maps):
    names = ('source_clip_id', 'clip_id', 'clip_uid', 'source_id', 'sequence_id', 'sequence_key')
    # Identity keys are ranked: the first one present in the NPZ or its sidecar
    # is the source identity.  HOT3D sidecars carry both source_clip_id
    # (the source clip) and clip_id (the window), which are legitimately
    # different; the lower-ranked keys are not competing candidates.
    found = []
    with np.load(path, allow_pickle=False) as z:
        for name in names:
            if name in z and z[name].size == 1:
                value = z[name].item()
                found.append(value.decode() if isinstance(value, bytes) else str(value))
                break
    if not found:
        for name in names:
            if metadata.get(name) is not None:
                found.append(str(metadata[name])); break
    mapped = maps.get(str(path), maps.get(str(path.resolve()), maps.get(path.name)))
    if mapped is not None: found.append(str(mapped))
    if not found and entry.get('allow_hot3d_clip_filename'):
        candidates = set(re.findall(r'(?<![A-Za-z0-9])clip-\d+(?!\d)', path.stem))
        if len(candidates) == 1: found = list(candidates)
    found = [s for s in found if s and s != 'None']
    if not found or len(set(found)) != 1:
        raise ValueError(f'Missing/conflicting source identity for {path}: {found}; provide an audited source map')
    return found[0]


def scan(spec):
    entries = spec['datasets']
    if sorted(e['dataset'] for e in entries) != sorted(DATASETS):
        raise ValueError('Exactly hot3d, oakink2 and phystwin are required')
    periods = [float(e['step_seconds']) for e in entries]
    if any(not math.isclose(t, 1/15, rel_tol=1e-5) for t in periods):
        raise ValueError('Use native H35/15Hz exports; do not silently reinterpret cadence')
    records, inventory = [], {}; all_paths = set()
    for entry in entries:
        label = entry['dataset']
        roots = [Path(s).expanduser().resolve() for s in entry['roots']]
        pinned_roots = {str(Path(s).expanduser().resolve()) for s in entry.get('holdout_roots', [])}
        protected_roots = {str(Path(s).expanduser().resolve()) for s in entry.get('protect_train_roots', [])}
        maps = dict(entry.get('source_map', {}))
        for filename in entry.get('source_map_files', []):
            if Path(filename).expanduser().is_file():
                for key, value in json.loads(Path(filename).expanduser().read_text()).items():
                    if key in maps and maps[key] != value: raise ValueError(f'Conflicting source map: {key}')
                    maps[key] = value
        rows, by_window = [], {}; pinned, protected = set(), set()
        for root in roots:
            paths = sorted(root.glob('*.npz'))
            if not root.is_dir() or not paths: raise ValueError(f'Processed corpus missing/empty: {root}')
            inventory[str(root)] = [str(p.absolute()) for p in paths]
            for lexical in paths:
                path = lexical.resolve(strict=True)
                side = lexical.with_suffix('.json')
                if not side.exists(): side = path.with_suffix('.json')
                metadata = json.loads(side.read_text()) if side.is_file() else {}
                source = _source(lexical, entry, metadata, maps)
                if str(root) in pinned_roots: pinned.add(source)
                if str(root) in protected_roots: protected.add(source)
                if path in all_paths:
                    if lexical.stem not in by_window or by_window[lexical.stem]['path'] != str(path):
                        raise ValueError(f'Physical file has multiple dataset/window identities: {path}')
                    continue
                if lexical.stem in by_window: raise ValueError(f'Duplicate window basename with different files: {lexical}')
                sx, hx = shape(path, 'scene_flows'), shape(path, 'robot_flows')
                dino = shape(path, 'scene_dino_features')
                with zipfile.ZipFile(path) as archive: keys = set(archive.namelist())
                if sx[0] != 35 or hx[0] != 35 or sx[-1] != 3 or hx[-1] != 3:
                    raise ValueError(f'{path}: require native H35 point/action trajectories')
                if len(dino) not in (2,3) or dino[-2] != sx[1] or dino[-1] != 768:
                    raise ValueError(f'{path}: require aligned 768-D DINO context features')
                if not {'scene_object_id.npy','scene_canonical_xyz.npy'} <= keys:
                    raise ValueError(f'{path}: full dexplicit_3d requires existing object/canonical fields')
                if label == 'phystwin' and not {'scene_transition_valid.npy','action_is_unlabelled.npy'} <= keys:
                    raise ValueError(f'{path}: use completed tracked-control PhysTwin export')
                frames = metadata.get('source_frame_indices', metadata.get('rgb_frame_ids'))
                fps = metadata.get('sampling', {}).get('source_fps', metadata.get('source_fps'))
                if frames is not None and fps:
                    dt = np.diff(np.asarray(frames, dtype=float))/float(fps)
                    if len(dt) != 34 or not np.allclose(dt, periods[0], rtol=1e-5, atol=1e-7):
                        raise ValueError(f'{path}: timestamps disagree with 15Hz')
                stat = path.stat()
                row = dict(dataset=label, source=source, window=lexical.stem, path=str(path),
                           size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                           metadata=str(side.resolve()) if side.is_file() else None,
                           metadata_sha256=digest(side) if side.is_file() else None)
                rows.append(row); by_window[lexical.stem] = row; all_paths.add(path)
        train, hold = split_sources({r['source'] for r in rows}, pinned, protected, seed=spec.get('seed',2026))
        for row in rows: row['role'] = 'train' if row['source'] in train else 'holdout'
        records.extend(rows)
    return sorted(records, key=lambda r:(r['dataset'],r['window'])), inventory


def summary(records):
    result = {}
    for name in DATASETS:
        rows = [r for r in records if r['dataset'] == name]
        held = [r for r in rows if r['role'] == 'holdout']
        result[name] = {'total_windows': len(rows), 'train_windows': len(rows)-len(held),
            'holdout_windows': len(held), 'total_sources': len({r['source'] for r in rows}),
            'holdout_sources': len({r['source'] for r in held}), 'window_holdout_fraction': len(held)/len(rows)}
    return result


def schedule(windows, epochs=50, batch=16):
    if windows < batch or epochs < 5 or epochs % 5:
        raise ValueError('Require >=one batch and a positive multiple of five epochs')
    steps = math.ceil(windows/batch)
    # Short budgets (<= 10 epochs) evaluate the full holdout every epoch; the
    # original 50-epoch recipe evaluates it every five epochs.
    full_every = steps if epochs <= 10 else 5*steps
    return dict(epochs=epochs, batch_size=batch, windows=windows, steps_per_epoch=steps,
                total_steps=steps*epochs, phase_a_steps=steps*2, warmup_steps=20,
                eval_every_steps=steps, full_every_steps=full_every, tail_padding=steps*batch-windows)


def quick_windows(records, cap=128, seed=2026):
    """Round-robin source-stratified fixed panel; contains every dataset."""
    chosen = []
    for dataset in DATASETS:
        groups = {}
        for r in records:
            if r['dataset'] == dataset and r['role'] == 'holdout': groups.setdefault(r['source'], []).append(r)
        ordered = sorted(groups, key=lambda s: hashlib.sha256(f'{seed}:{dataset}:{s}'.encode()).hexdigest())
        for source in groups:
            groups[source].sort(key=lambda r: hashlib.sha256(f'{seed}:{r["window"]}'.encode()).hexdigest())
        panel = []; depth = 0
        while len(panel) < cap:
            added = [groups[s][depth] for s in ordered if depth < len(groups[s])]
            if not added: break
            panel.extend(added[:cap-len(panel)]); depth += 1
        if not panel: raise ValueError(f'No held-out windows for {dataset}')
        chosen.extend((r['dataset'],r['window']) for r in panel)
    return chosen


def materialize_views(root, records):
    """Create symlink views only; never alter original files/splits."""
    root = Path(root)
    for row in records:
        dest = root/'data'/row['role']/row['dataset']/(row['window']+'.npz')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            if dest.resolve() != Path(row['path']): raise ValueError(f'View changed: {dest}')
        elif dest.exists(): raise ValueError(f'Refusing to replace existing file: {dest}')
        else: dest.symlink_to(row['path'])
        if row['metadata']:
            meta = dest.with_suffix('.json')
            if not meta.exists(): meta.symlink_to(row['metadata'])


def verify_records(plan):
    for root, expected in plan['inventory'].items():
        if sorted(str(p.absolute()) for p in Path(root).glob('*.npz')) != expected:
            raise ValueError(f'Processed inventory changed: {root}; create a new campaign')
    for row in plan['records']:
        stat = Path(row['path']).stat()
        if stat.st_size != row['size'] or stat.st_mtime_ns != row['mtime_ns']:
            raise ValueError(f'Dataset file changed: {row["path"]}')
        if row['metadata'] and digest(row['metadata']) != row['metadata_sha256']:
            raise ValueError(f'Source metadata changed: {row["metadata"]}')
