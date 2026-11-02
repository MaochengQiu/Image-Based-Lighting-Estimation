"""Dataset loading, configuration identity and checkpoint helpers."""
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import read_hdr


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path, size):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Explicit resize, no hidden crop that would change the camera's field of view.
    image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1).copy()) / 255


class LightingDataset(Dataset):
    def __init__(self, manifest, split, config, unique_targets=False):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        preparation = json.loads((self.root / 'preparation.json').read_text(encoding='utf-8'))
        if preparation['config']['data'] != config['data']:
            raise ValueError('Data configuration differs from preparation.json; regenerate explicitly')
        all_rows = [json.loads(line) for line in self.manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
        scenes, targets, ids = {}, {}, set()
        for row in all_rows:
            if row['sample_id'] in ids:
                raise ValueError(f"Duplicate sample id: {row['sample_id']}")
            ids.add(row['sample_id'])
            for field, registry in [('scene_id', scenes), ('target_path', targets)]:
                key = row[field]
                if registry.setdefault(key, row['split']) != row['split']:
                    raise ValueError(f"Split leakage in {field}: {key}")
        self.rows = [r for r in all_rows if r['split'] == split]
        if unique_targets:
            self.rows = list({r['target_path']: r for r in self.rows}.values())
        if not self.rows:
            raise ValueError(f"No samples in split {split}")
        self.config = config
        self.fingerprint = sha256_file(self.manifest)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = load_rgb(self.root / row['input_path'], self.config['data']['input_size'])
        target = read_hdr(self.root / row['target_path'])
        if list(target.shape[:2]) != self.config['data']['env_size']:
            raise ValueError(f"Unexpected target shape: {target.shape}")
        return {'image': image, 'hdr': torch.from_numpy(target.transpose(2, 0, 1).copy()),
                'sample_id': row['sample_id']}


def load_checkpoint(path, device='cpu'):
    # Only load local/trusted checkpoints. This project writes tensors and primitive dictionaries.
    return torch.load(path, map_location=device, weights_only=True)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
