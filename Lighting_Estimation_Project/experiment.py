"""Real-data preparation, two-stage training, evaluation and single-image inference."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.lighting.data import prepare, preview, write_rgb
from src.lighting.models import AutoregressiveLighting, Tokenizer
from src.lighting.objectives import IlluminationObjective
from src.lighting.runtime import (LightingDataset, load_checkpoint, load_rgb,
                                  save_json, seed_everything, sha256_file)


def device_for(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def mean_metrics(rows):
    if not rows:
        raise ValueError('Cannot aggregate an empty evaluation')
    return {key: sum(r[key] for r in rows) / len(rows) for key in rows[0] if key != 'sample_id'}


def assert_finite(loss):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite objective; stopping rather than recording invalid results')


def save_preview(out_dir, sample_id, image, target, predicted_log, objective):
    prediction = objective.to_hdr(predicted_log)
    def rgb(tensor):
        return tensor.detach().cpu().permute(1, 2, 0).numpy()
    write_rgb(out_dir / f'{sample_id}_input.png', rgb(image[0]))
    for label, value in [('target', target), ('predicted', prediction)]:
        write_rgb(out_dir / f'{sample_id}_{label}.hdr', rgb(value[0]), hdr=True)
        write_rgb(out_dir / f'{sample_id}_{label}.png', preview(rgb(value[0])))
        diffuse, glossy = objective.render_spheres(value)
        write_rgb(out_dir / f'{sample_id}_{label}_diffuse.png', preview(rgb(diffuse[0])))
        write_rgb(out_dir / f'{sample_id}_{label}_glossy.png', preview(rgb(glossy[0])))


@torch.no_grad()
def validate(model, tokenizer, loader, objective, device, stage, limit=0, out_dir=None):
    model.eval()
    rows = []
    histogram = torch.zeros(tokenizer.codebook_size, dtype=torch.long, device=device)
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.synchronize()
        started = time.perf_counter()
    for batch in loader:
        image, target = batch['image'].to(device), batch['hdr'].to(device)
        if limit:
            remaining = limit - len(rows)
            if remaining <= 0:
                break
            image, target = image[:remaining], target[:remaining]
        if stage == 'tokenizer':
            prediction, tokens, _ = model(torch.log1p(target / objective.scale))
        else:
            # No teacher forcing and no target tokens in this branch.
            tokens = model.generate(image)
            prediction = tokenizer.decode(tokens)
        histogram += torch.bincount(tokens.flatten(), minlength=tokenizer.codebook_size)
        metrics = objective.metrics(prediction, target)
        for index, values in enumerate(metrics):
            rows.append({'sample_id': batch['sample_id'][index], **values})
        if out_dir is not None and len(rows) <= 8:
            save_preview(out_dir, batch['sample_id'][0], image[:1], target[:1], prediction[:1], objective)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    summary = mean_metrics(rows)
    frequency = histogram.float() / histogram.sum().clamp_min(1)
    summary.update({'samples': len(rows), 'seconds_including_metrics_and_previews': elapsed,
                    'active_codes': int((histogram > 0).sum()),
                    'code_perplexity': float(torch.exp(-(frequency * frequency.clamp_min(1e-12).log()).sum()))})
    return rows, summary


def train(args, config, device):
    stage = args.stage
    output = Path(args.out_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Use an empty output directory or --resume; existing results will not be overwritten')
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(config).to(device)
    tokenizer_hash = None
    if stage == 'ar':
        if not args.tokenizer:
            raise ValueError('--tokenizer is required for AR training')
        saved = load_checkpoint(args.tokenizer, device)
        if saved['stage'] != 'tokenizer' or saved['config']['data'] != config['data'] or saved['config']['tokenizer'] != config['tokenizer']:
            raise ValueError('Incompatible tokenizer checkpoint')
        tokenizer.load_state_dict(saved['model'])
        tokenizer.requires_grad_(False).eval()
        tokenizer_hash = sha256_file(args.tokenizer)
        model = AutoregressiveLighting(config, pretrained=(not args.resume and config['ar']['pretrained_image_encoder'])).to(device)
    else:
        model = tokenizer
    unique = stage == 'tokenizer'
    train_set = LightingDataset(args.manifest, 'train', config, unique_targets=unique)
    val_set = LightingDataset(args.manifest, 'val', config, unique_targets=unique)
    settings = config['training']
    loader = DataLoader(train_set, batch_size=settings['batch_size'], shuffle=True,
                        num_workers=settings['num_workers'], pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_set, batch_size=settings['batch_size'], shuffle=False,
                            num_workers=settings['num_workers'])
    objective = IlluminationObjective(config).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=settings['tokenizer_lr' if unique else 'ar_lr'])
    epoch_start, best = 0, float('inf')
    if args.resume:
        saved = load_checkpoint(args.resume, device)
        if (saved['stage'] != stage or saved['config'] != config or
                saved['manifest_sha256'] != train_set.fingerprint or saved['tokenizer_sha256'] != tokenizer_hash):
            raise ValueError('Resume config, dataset, stage or tokenizer differs from saved run')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        epoch_start, best = saved['epoch'] + 1, saved['best_log_rmse']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type == 'cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    save_json(output / 'config.json', config)
    save_json(output / 'environment.json', {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                                           'device': str(device), 'manifest_sha256': train_set.fingerprint,
                                           'tokenizer_sha256': tokenizer_hash})
    print(f'{stage}: real paired HDR training; validation uses independent scenes and complete generation.', flush=True)
    accumulation = settings['accumulation_steps']
    if accumulation < 1:
        raise ValueError('accumulation_steps must be positive')
    for epoch in range(epoch_start, settings['epochs']):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count = 0.0, 0
        for step, batch in enumerate(loader):
            target, image = batch['hdr'].to(device), batch['image'].to(device)
            log_target = torch.log1p(target / objective.scale)
            if unique:
                prediction, _, vq_loss = tokenizer(log_target)
                loss = objective(prediction, target) + vq_loss
            else:
                with torch.no_grad():
                    tokens = tokenizer.encode(log_target)
                logits = model(image, tokens)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.reshape(-1))
            assert_finite(loss)
            # Exact sample weighting also handles the last incomplete microbatch group.
            group_start = (step // accumulation) * accumulation * settings['batch_size']
            group_count = min(accumulation * settings['batch_size'], len(train_set) - group_start)
            (loss * target.shape[0] / group_count).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * target.shape[0]
            count += target.shape[0]
        _, metrics = validate(model, tokenizer, val_loader, objective, device, stage,
                              limit=settings['validation_generation_limit'])
        improved = metrics['log_rmse'] < best
        best = min(best, metrics['log_rmse'])
        state = {'stage': stage, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'config': config, 'epoch': epoch, 'best_log_rmse': best,
                 'manifest_sha256': train_set.fingerprint, 'tokenizer_sha256': tokenizer_hash,
                 'tokenizer_path': str(Path(args.tokenizer).resolve()) if args.tokenizer else None,
                 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
        torch.save(state, output / 'last.pt')
        if improved:
            torch.save(state, output / 'best.pt')
        record = {'epoch': epoch, 'train_objective': total / count, 'validation': metrics}
        with (output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record), flush=True)
        with (output / 'notes.txt').open('a', encoding='utf-8') as stream:
            stream.write(f"Epoch {epoch}: validation log RMSE {metrics['log_rmse']:.6g}; no test-set selection.\n")
    if epoch_start >= settings['epochs']:
        raise ValueError('Checkpoint has already reached configured epochs')


def restore(args, device):
    saved = load_checkpoint(args.checkpoint, device)
    config, stage = saved['config'], saved['stage']
    tokenizer = Tokenizer(config).to(device)
    if stage == 'tokenizer':
        tokenizer.load_state_dict(saved['model'])
        model = tokenizer
    elif stage == 'ar':
        path = args.tokenizer or saved['tokenizer_path']
        if sha256_file(path) != saved['tokenizer_sha256']:
            raise ValueError('Tokenizer hash differs from the AR training checkpoint')
        tokenizer.load_state_dict(load_checkpoint(path, device)['model'])
        model = AutoregressiveLighting(config, pretrained=False).to(device)
        model.load_state_dict(saved['model'])
    else:
        raise ValueError(f'Unexpected checkpoint stage: {stage}')
    tokenizer.eval()
    model.eval()
    return config, stage, model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['prepare', 'tokenizer', 'ar', 'evaluate', 'predict'])
    parser.add_argument('--config', default='configs/initial.json')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--sources', help='CSV with scene_id,pano_path[,split]')
    parser.add_argument('--manifest', help='Prepared manifest.jsonl')
    parser.add_argument('--tokenizer', help='Tokenizer checkpoint')
    parser.add_argument('--checkpoint', help='Checkpoint for evaluation/inference')
    parser.add_argument('--resume', help='Resume an identical training run')
    parser.add_argument('--input', help='LDR photo for inference, matching the configured FOV/aspect')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    seed_everything(config['seed'])
    if args.stage == 'prepare':
        if not args.sources:
            parser.error('--sources is required')
        print(f"Prepared {prepare(args.sources, args.out_dir, config)} real-data samples")
        return
    device = device_for(args.device)
    if args.stage in {'tokenizer', 'ar'}:
        if not args.manifest:
            parser.error('--manifest is required')
        train(args, config, device)
        return
    if not args.checkpoint:
        parser.error('--checkpoint is required')
    config, stage, model, tokenizer = restore(args, device)
    objective = IlluminationObjective(config).to(device)
    output = Path(args.out_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Evaluation/inference requires an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'predict':
        if stage != 'ar' or not args.input:
            parser.error('Prediction requires an AR checkpoint and --input')
        image = load_rgb(args.input, config['data']['input_size'])[None].to(device)
        with torch.no_grad():
            prediction = objective.to_hdr(tokenizer.decode(model.generate(image)))
        rgb = prediction[0].cpu().permute(1, 2, 0).numpy()
        write_rgb(output / 'prediction.hdr', rgb, hdr=True)
        write_rgb(output / 'prediction.png', preview(rgb))
        save_json(output / 'inference.json', {'checkpoint_sha256': sha256_file(args.checkpoint),
                                             'config': config, 'generation': 'greedy',
                                             'max_radiance': float(rgb.max()),
                                             'radiometric_calibration': 'source-relative; not lux'})
        return
    if not args.manifest:
        parser.error('--manifest is required')
    dataset = LightingDataset(args.manifest, args.split, config, unique_targets=stage == 'tokenizer')
    rows, summary = validate(model, tokenizer, DataLoader(dataset, batch_size=1), objective,
                             device, stage, out_dir=output)
    save_json(output / 'per_sample.json', rows)
    save_json(output / 'final_info.json', {'stage': stage, 'split': args.split, 'metrics': summary,
                                          'manifest_sha256': dataset.fingerprint,
                                          'checkpoint_sha256': sha256_file(args.checkpoint),
                                          'selection_metric': 'validation log_rmse',
                                          'generation': 'greedy' if stage == 'ar' else 'reconstruction'})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
