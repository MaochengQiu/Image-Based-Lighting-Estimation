"""Controlled AR experiments with a fixed dataset and tokenizer."""
import copy
import html
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import save_preview
from src.lighting.models import AutoregressiveLighting, Tokenizer
from src.lighting.objectives import IlluminationObjective
from src.lighting.runtime import LightingDataset, sha256_file

BASE = ROOT / "runs/demo_optimized_v1"
OUT = ROOT / "runs/ar_controlled"
MANIFEST = BASE / "prepared/manifest.jsonl"
TOKENIZER = BASE / "tokenizer/best.pt"


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def diagnose(path, directory):
    saved = read_checkpoint(path)
    config = saved["config"]
    model = AutoregressiveLighting(config)
    model.load_state_dict(saved["model"])
    model.eval()
    tokenizer = Tokenizer(config)
    tokenizer.load_state_dict(read_checkpoint(TOKENIZER)["model"])
    tokenizer.eval()
    objective = IlluminationObjective(config)
    train = LightingDataset(MANIFEST, "train", config)
    indices = [i for i, row in enumerate(train.rows)
               if row["yaw_degrees"] == 0 and row["exposure"] == 1.0]
    train_images = torch.stack([train[i]["image"] for i in indices])
    train_hdr = torch.stack([train[i]["hdr"] for i in indices])
    val = LightingDataset(MANIFEST, "val", config)
    val_images = torch.stack([val[i]["image"] for i in range(len(val))])
    val_hdr = torch.stack([val[i]["hdr"] for i in range(len(val))])
    with torch.inference_mode():
        target = tokenizer.encode(torch.log1p(train_hdr / objective.scale))
        logits = model(train_images, target)
        labels = model.sequence_targets(target)
        # Identical correct prefixes must give identical full-forward and step logits.
        image, tokens = train_images[:1], target[:1]
        full = model(image, tokens)
        memory = model.image_features(image)
        prior = model.condition_logits(memory) if model.use_condition_prior else None
        sequence = model.sequence_targets(tokens)
        max_error = 0.0
        for length in (1, 2, 8, 16, 32, 64, model.length):
            prefix = torch.cat([torch.full((1, 1), model.bos, dtype=torch.long),
                                sequence[:, :length-1]], 1)
            step = model.next_logits(prefix, memory)[:, -1]
            if prior is not None:
                step = step + prior[:, length-1]
            max_error = max(max_error, float((step-full[:, length-1]).abs().max()))
        if max_error > 0.0001:
            raise AssertionError(f"Training/inference mismatch: {max_error}")
        generated = model.generate(train_images)
        val_generated = model.generate(val_images)
        swapped = model.generate(train_images.roll(1, 0))
        val_swapped = model.generate(val_images.flip(0))
        prediction = tokenizer.decode(generated)
        val_prediction = tokenizer.decode(val_generated)
        train_scores = objective.metrics(prediction, train_hdr)
        val_scores = objective.metrics(val_prediction, val_hdr)
        first_errors = []
        mismatch = model.sequence_targets(generated) != labels
        for row in mismatch:
            indices_wrong = row.nonzero()
            first_errors.append(int(indices_wrong[0]) if len(indices_wrong) else model.length)
        data = {
            "epoch": saved["epoch"],
            "prefix_logits_max_error": max_error,
            "teacher_token_accuracy": float((logits.argmax(-1) == labels).float().mean()),
            "free_token_accuracy": float((generated == target).float().mean()),
            "first_error_positions_train": first_errors,
            "train_log_rmse": sum(row["log_rmse"] for row in train_scores) / len(train_scores),
            "val_log_rmse": sum(row["log_rmse"] for row in val_scores) / len(val_scores),
            "val_diffuse_rmse": sum(row["diffuse_rmse"] for row in val_scores) / len(val_scores),
            "cross_scene_changed_tokens": float((generated != swapped).float().mean()),
            "val_yaw_changed_tokens": float((val_generated != val_swapped).float().mean()),
            "checkpoint_sha256": sha256_file(path),
            "manifest_sha256": sha256_file(MANIFEST),
            "tokenizer_sha256": sha256_file(TOKENIZER)
        }
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(len(val)):
            save_preview(directory, val.rows[index]["sample_id"], val_images[index:index+1],
                         val_hdr[index:index+1], val_prediction[index:index+1], objective)
    (directory / "diagnostic.json").write_text(json.dumps(data, indent=2))
    return data


def main():
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ("best", "last"):
        directory = OUT / "baseline" / name
        if (directory / "diagnostic.json").exists():
            results["baseline_" + name] = json.loads((directory / "diagnostic.json").read_text())
        else:
            results["baseline_" + name] = diagnose(BASE / "ar_condition" / (name + ".pt"), directory)
    original = json.loads((BASE / "config_condition.json").read_text())
    variants = {
        "visible_first": {"generation_order": "visible_first"},
        "prefix_noise": {"prefix_noise_probability": 0.1},
        "pretrained": {"image_encoder": "resnet18", "pretrained_image_encoder": True,
                       "freeze_image_encoder": True, "encoder_input_size": [224, 224],
                       "cache_frozen_features": True}
    }
    for name, changes in variants.items():
        directory = OUT / name
        config = copy.deepcopy(original)
        config["ar"].update(changes)
        config_file = OUT / (name + ".json")
        config_file.write_text(json.dumps(config, indent=2))
        last = directory / "last.pt"
        complete = last.exists() and read_checkpoint(last)["epoch"] + 1 >= config["training"]["epochs"]
        if not complete:
            arguments = [sys.executable, "-u", str(ROOT / "experiment.py"),
                         "--stage", "ar", "--config", str(config_file), "--manifest", str(MANIFEST),
                         "--tokenizer", str(TOKENIZER), "--out_dir", str(directory), "--device", "cpu"]
            if last.exists():
                arguments.extend(["--resume", str(last)])
            print("Training", name, flush=True)
            with (OUT / (name + ".log")).open("a", encoding="utf-8") as stream:
                subprocess.run(arguments, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        for checkpoint in ("best", "last"):
            preview = directory / ("diagnostic_" + checkpoint)
            if (preview / "diagnostic.json").exists():
                metrics = json.loads((preview / "diagnostic.json").read_text())
            else:
                metrics = diagnose(directory / (checkpoint + ".pt"), preview)
            results[name + "_" + checkpoint] = metrics
            print(name, checkpoint, json.dumps(metrics), flush=True)
        (OUT / "results_partial.json").write_text(json.dumps(results, indent=2))
    # Select among configurations by validation only, then describe the test result once.
    candidates = {name: metrics for name, metrics in results.items() if name.endswith("_best")}
    responsive = {name: metrics for name, metrics in candidates.items()
                  if metrics['val_yaw_changed_tokens'] > 0 and metrics['cross_scene_changed_tokens'] > 0}
    pool = responsive or candidates
    selected = min(pool, key=lambda name: pool[name]['val_log_rmse'])
    selected_name = selected[:-5]
    checkpoint = BASE / "ar_condition/best.pt" if selected_name == "baseline" else OUT / selected_name / "best.pt"
    test_directory = OUT / ('selected_test_' + selected_name)
    if not (test_directory / 'final_info.json').exists():
        subprocess.run([sys.executable, str(ROOT / "experiment.py"), "--stage", "evaluate",
                        "--manifest", str(MANIFEST), "--checkpoint", str(checkpoint),
                        "--tokenizer", str(TOKENIZER), "--out_dir", str(OUT / ("selected_test_" + selected_name)),
                        "--device", "cpu"], cwd=ROOT, check=True)
    result = {"runs": results, "selected_by_validation": selected,
              "selection_rule": "Require nonconstant generated response to changed validation/train inputs, then minimize validation log RMSE. This checks responsiveness, not correctness.",
              "test": json.loads((test_directory / "final_info.json").read_text()),
              "limits": "One seed, eight training scenes, one validation and one test scene."}
    (OUT / "results.json").write_text(json.dumps(result, indent=2))
    rows = []
    for name, metrics in results.items():
        rows.append("<tr><td>" + name + "</td>" + "".join(
            f"<td>{metrics[key]:.4f}</td>" for key in (
                "teacher_token_accuracy", "free_token_accuracy", "train_log_rmse",
                "val_log_rmse", "cross_scene_changed_tokens")) + "</tr>")
    (OUT / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Controlled autoregressive experiments</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:16px sans-serif}td,th{padding:10px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'
        '<h1>Controlled autoregressive experiments</h1><p>Fixed dataset, tokenizer, seed and 100 epochs. '
        'One change per configuration. Pretrained ResNet includes its input resizing and frozen feature cache.</p>'
        '<table><tr><th>Run</th><th>Teacher accuracy</th><th>Free accuracy</th><th>Train log RMSE</th>'
        '<th>Validation log RMSE</th><th>Changed tokens across scenes</th></tr>' + "".join(rows) +
        '</table><p>Selected by validation: ' + html.escape(selected) + '</p><pre>' +
        html.escape(json.dumps(result, indent=2)) + '</pre>', encoding="utf-8")
    print("Completed", OUT / "index.html", flush=True)


if __name__ == "__main__":
    main()
