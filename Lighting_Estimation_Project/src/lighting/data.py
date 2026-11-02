"""HDR contracts and projection: +X right, +Y up, +Z forward.

ERP longitude is [-pi, pi], latitude [pi/2, -pi/2], at pixel centres.
Only yaw varies in this first camera-centred implementation.
"""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def read_hdr(path):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        image = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {'.hdr', '.exr'}:
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"HDR decode failed: {path}; EXR may require an additional codec")
        image = image[..., ::-1]
    else:
        raise ValueError(f"Expected linear RGB .hdr/.exr/.npy, not {path.suffix}")
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[1] != 2 * image.shape[0]:
        raise ValueError(f"Expected H x 2H x 3 ERP: {path}, got {image.shape}")
    if not np.isfinite(image).all() or (image < 0).any():
        raise ValueError(f"Nonfinite or negative radiance in {path}; repair source explicitly")
    return image


def write_rgb(path, image, hdr=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image, dtype=np.float32)
    if not np.isfinite(image).all() or (image < 0).any():
        raise ValueError('Output must be finite and nonnegative')
    image = image if hdr else np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    ok, encoded = cv2.imencode('.hdr' if hdr else '.png', image[..., ::-1])
    if not ok:
        raise ValueError(f"Could not encode {path}")
    encoded.tofile(path)


def srgb(linear):
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * linear ** (1 / 2.4) - 0.055).astype(np.float32)


def preview(hdr):
    return srgb(hdr / (1 + hdr))


def erp_directions(height, width):
    lon = ((np.arange(width) + 0.5) / width - 0.5) * 2 * np.pi
    lat = (0.5 - (np.arange(height) + 0.5) / height) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([np.cos(lat) * np.sin(lon), np.sin(lat),
                     np.cos(lat) * np.cos(lon)], axis=-1).astype(np.float32)


def sample_erp(image, rays, yaw_degrees):
    yaw = np.deg2rad(yaw_degrees)
    x, y, z = np.moveaxis(rays, -1, 0)
    world_x = np.cos(yaw) * x + np.sin(yaw) * z
    world_z = -np.sin(yaw) * x + np.cos(yaw) * z
    lon = np.arctan2(world_x, world_z)
    lat = np.arctan2(y, np.sqrt(world_x ** 2 + world_z ** 2))
    h, w = image.shape[:2]
    map_x = ((lon / (2 * np.pi) + 0.5) * w - 0.5 + 1).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * h - 0.5).astype(np.float32)
    # Wrap longitude with a halo, while clamping latitude at the poles.
    wrapped = np.concatenate([image[:, -1:], image, image[:, :1]], axis=1)
    return cv2.remap(wrapped, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def perspective(image, size, hfov_degrees, yaw_degrees):
    height, width = size
    if not 0 < hfov_degrees < 180:
        raise ValueError('Horizontal FOV must be between 0 and 180 degrees')
    focal = width / (2 * np.tan(np.deg2rad(hfov_degrees) / 2))
    x, y = np.meshgrid((np.arange(width) + 0.5 - width / 2) / focal,
                       (height / 2 - np.arange(height) - 0.5) / focal)
    return sample_erp(image, np.stack([x, y, np.ones_like(x)], -1), yaw_degrees)


def resize_radiance(image, size):
    h = image.shape[0]
    weights = np.sin(np.pi * (np.arange(h) + 0.5) / h).astype(np.float32)[:, None]
    weights = np.broadcast_to(weights, image.shape[:2]).copy()
    width, height = size[1], size[0]
    weighted = cv2.resize(image * weights[..., None], (width, height), interpolation=cv2.INTER_AREA)
    area = cv2.resize(weights, (width, height), interpolation=cv2.INTER_AREA)
    return weighted / area[..., None]


def prepare(source_csv, out_dir, config):
    source_csv = Path(source_csv).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"Preparation requires an empty output directory: {out_dir}")
    with source_csv.open(encoding='utf-8-sig', newline='') as stream:
        sources = list(csv.DictReader(stream))
    if not sources or any(not r.get('scene_id') or not r.get('pano_path') for r in sources):
        raise ValueError('Source CSV requires scene_id,pano_path; optional split')
    groups = sorted({r['scene_id'] for r in sources})
    explicit = [bool(r.get('split')) for r in sources]
    if any(explicit) and not all(explicit):
        raise ValueError('Provide split on every row or on none of them')
    if all(explicit):
        group_split = {}
        for row in sources:
            split = row['split']
            if split not in {'train', 'val', 'test'}:
                raise ValueError(f"Unknown split: {split}")
            previous = group_split.setdefault(row['scene_id'], split)
            if previous != split:
                raise ValueError(f"Scene leakage: {row['scene_id']}")
    else:
        if len(groups) < 3:
            raise ValueError('Need at least three independent scenes for train/val/test')
        np.random.default_rng(config['seed']).shuffle(groups)
        fractions = config['data']['split_fractions']
        if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
            raise ValueError('split_fractions must be three positive values summing to one')
        n_val = max(1, round(len(groups) * fractions[1]))
        n_test = max(1, round(len(groups) * fractions[2]))
        if n_val + n_test >= len(groups):
            raise ValueError('Split fractions leave no training scenes')
        group_split = {g: ('val' if i < n_val else 'test' if i < n_val + n_test else 'train')
                       for i, g in enumerate(groups)}
    if set(group_split.values()) != {'train', 'val', 'test'}:
        raise ValueError('All three splits must contain scenes')
    data = config['data']
    scale = data['hdr_scale']
    if scale <= 0 or not data['yaws_degrees'] or not data['train_exposures'] or min(data['train_exposures']) <= 0:
        raise ValueError('HDR scale/exposures must be positive; yaw list must be nonempty')
    records, audit, fingerprints = [], [], {}
    for row in sources:
        path = (source_csv.parent / row['pano_path']).resolve()
        image = read_hdr(path)
        if np.log1p(image.max() / scale) > data['max_log_value']:
            raise ValueError(f"Source exceeds configured max_log_value: {path}")
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest in fingerprints:
            raise ValueError(f"Duplicate panorama: {path} and {fingerprints[digest]}")
        fingerprints[digest] = str(path)
        row['_path'] = str(path)
        audit.append({'path': str(path), 'sha256_rgb_float32': digest, 'scene_id': row['scene_id'],
                      'split': group_split[row['scene_id']], 'shape': list(image.shape),
                      'min': float(image.min()), 'max': float(image.max())})
    out_dir.mkdir(parents=True, exist_ok=True)
    for source_index, row in enumerate(sources):
        image = read_hdr(row['_path'])
        split = group_split[row['scene_id']]
        small = resize_radiance(image, data['env_size'])
        for view_index, yaw in enumerate(data['yaws_degrees']):
            stem = f"p{source_index:06d}_v{view_index:03d}"
            target = sample_erp(small, erp_directions(*data['env_size']), yaw)
            target_path = f"targets/{stem}.npy"
            (out_dir / 'targets').mkdir(exist_ok=True)
            np.save(out_dir / target_path, target)
            crop = perspective(image, data['input_size'], data['hfov_degrees'], yaw)
            exposures = data['train_exposures'] if split == 'train' else [1.0]
            write_rgb(out_dir / f"previews/{stem}.png", preview(target))
            for exposure_index, exposure in enumerate(exposures):
                sample_id = f"{stem}_e{exposure_index}"
                input_path = f"inputs/{sample_id}.png"
                write_rgb(out_dir / input_path, srgb(crop * exposure))
                records.append({'sample_id': sample_id, 'scene_id': row['scene_id'],
                                'split': split, 'input_path': input_path, 'target_path': target_path,
                                'yaw_degrees': yaw, 'hfov_degrees': data['hfov_degrees'],
                                'pitch_degrees': 0, 'roll_degrees': 0, 'exposure': exposure,
                                'response': 'linear_clip_then_srgb', 'projection_version': 1})
    with (out_dir / 'manifest.jsonl').open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    (out_dir / 'preparation.json').write_text(json.dumps(
        {'config': config, 'scene_split': group_split, 'sources': audit,
         'sample_counts': {s: sum(r['split'] == s for r in records) for s in ['train', 'val', 'test']},
         'radiometric_calibration': 'unverified; values follow source files'},
        ensure_ascii=False, indent=2), encoding='utf-8')
    return len(records)
