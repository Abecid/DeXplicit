"""Download the selected release checkpoints from Hugging Face and verify SHA-256."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''): h.update(part)
    return h.hexdigest()


def main():
    manifest = json.loads((Path(__file__).parent/'resources/checkpoints.json').read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['all', *manifest['models']], default='all')
    p.add_argument('--output-dir', default='weights'); a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    keys = manifest['default_models'] if a.model == 'all' else [a.model]
    from huggingface_hub import hf_hub_download
    for key in keys:
        item = manifest['models'][key]; target = out/item['file']
        if target.exists():
            if digest(target) != item['sha256']: raise ValueError(f'Existing file has a different checksum: {target}')
            print(f'Already verified: {target}'); continue
        cached = hf_hub_download(manifest['huggingface_repo'], item['file'], revision=manifest['huggingface_revision'])
        if Path(cached).stat().st_size != item['bytes'] or digest(cached) != item['sha256']:
            raise ValueError('Downloaded checkpoint does not match the release manifest')
        temp = target.with_suffix('.partial')
        try:
            shutil.copyfile(cached, temp); os.replace(temp, target)
        finally:
            if temp.exists(): temp.unlink()
        print(f'Verified: {target}')

if __name__ == '__main__': main()
