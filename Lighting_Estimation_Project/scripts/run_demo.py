"""Run a small real-HDR engineering demo, not a benchmark or pretrained estimator."""
import argparse
import csv
import hashlib
import html
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = [("small_empty_room_1", "train"), ("lebombo", "val"), ("studio_small_08", "test")]


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        "User-Agent": "LightingEstimationResearchDemo/0.1"}), timeout=90)


def acquire():
    raw = ROOT / "data/raw/polyhaven_demo"
    raw.mkdir(parents=True, exist_ok=True)
    cached = ROOT / "data/manifests/polyhaven_demo/provenance.json"
    source_file = cached.parent / "sources.csv"
    if cached.exists() and source_file.exists():
        old = json.loads(cached.read_text())
        if [(a["asset"], a["split"]) for a in old["assets"]] == ASSETS and all(
            (raw / (a["asset"] + ".hdr")).exists() and
            hashlib.md5((raw / (a["asset"] + ".hdr")).read_bytes()).hexdigest() == a["md5"]
            for a in old["assets"]):
            return source_file
    sources, provenance = [], []
    for asset, split in ASSETS:
        with request("https://api.polyhaven.com/files/" + asset) as stream:
            file = json.load(stream)["hdri"]["1k"]["hdr"]
        target = raw / (asset + ".hdr")
        if not target.exists():
            with request(file["url"]) as stream:
                payload = stream.read()
            if len(payload) != file["size"] or hashlib.md5(payload).hexdigest() != file["md5"]:
                raise ValueError("Download checksum mismatch: " + asset)
            target.write_bytes(payload)
        payload = target.read_bytes()
        if hashlib.md5(payload).hexdigest() != file["md5"]:
            raise ValueError("Existing file checksum mismatch: " + str(target))
        sources.append(dict(scene_id=asset, pano_path=str(target), split=split))
        provenance.append(dict(asset=asset, split=split, url=file["url"],
                               md5=file["md5"], sha256=hashlib.sha256(payload).hexdigest(),
                               license="CC0", page="https://polyhaven.com/a/" + asset))
    directory = ROOT / "data/manifests/polyhaven_demo"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "sources.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scene_id", "pano_path", "split"])
        writer.writeheader()
        writer.writerows(sources)
    (directory / "provenance.json").write_text(json.dumps({
        "purpose": "Three-scene engineering demo only; not a generalization benchmark",
        "calibration": "source-relative radiance; absolute photometry unverified",
        "license_url": "https://polyhaven.com/license", "assets": provenance}, indent=2))
    return source


def call(arguments, directory):
    command = [sys.executable, "-u", str(ROOT / "experiment.py"), *map(str, arguments)]
    print("RUN:", " ".join(command), flush=True)
    with (directory / "commands.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(command) + "\n")
    subprocess.run(command, cwd=ROOT, check=True)


def figures(directory, ar_folder="ar_eval"):
    import cv2
    import numpy as np
    panels = []
    for stage in ("tokenizer_eval", ar_folder):
        folder = directory / stage
        candidates = sorted(folder.glob("*_input.png"))
        if not candidates:
            raise ValueError("Evaluation produced no previews")
        prefix = candidates[0].name[:-len("_input.png")]
        rows = []
        for label, suffix in [("Input", "input"), ("Target HDR preview", "target"),
                              ("Predicted HDR preview", "predicted"),
                              ("Target diffuse sphere", "target_diffuse"),
                              ("Predicted diffuse sphere", "predicted_diffuse"),
                              ("Target glossy diagnostic", "target_glossy"),
                              ("Predicted glossy diagnostic", "predicted_glossy")]:
            image = cv2.imdecode(np.fromfile(folder / f"{prefix}_{suffix}.png", np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Missing preview: " + suffix)
            h, w = image.shape[:2]
            ratio = min(512/w, 180/h)
            image = cv2.resize(image, (round(w*ratio), round(h*ratio)))
            canvas = np.full((220, 560, 3), 245, np.uint8)
            cv2.putText(canvas, label, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (20,20,20), 1)
            canvas[36:36+image.shape[0], 16:16+image.shape[1]] = image
            rows.append(canvas)
        panel = np.concatenate(rows, axis=0)
        name = "Figure_1.png" if stage == "tokenizer_eval" else "Figure_2.png"
        ok, encoded = cv2.imencode(".png", panel)
        if not ok:
            raise ValueError("Figure encoding failed")
        encoded.tofile(directory / name)
        panels.append(name)
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT/"runs"/("demo_"+time.strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokenizer-epochs", type=int, default=40)
    parser.add_argument("--ar-epochs", type=int, default=30)
    args = parser.parse_args()
    if min(args.tokenizer_epochs,args.ar_epochs) < 1:
        parser.error("Epoch counts must be positive")
    import torch
    torch.set_num_threads(2)
    # Also bound the threads in subprocess training.
    import os
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty demo output directory")
    print("Three real HDRs, one train scene, one validation scene, one test scene. "
          "This run verifies code execution only; it cannot establish useful accuracy.", flush=True)
    import importlib.metadata
    (output/"environment-lock.txt").write_text("\n".join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())), encoding="utf-8")
    source = acquire()
    config = json.loads((ROOT/"configs/initial.json").read_text())
    config["data"].update(input_size=[64,64], env_size=[64,128])
    config["tokenizer"].update(base_channels=16, embedding_dim=16, codebook_size=32, initialize_from_data=True)
    config["ar"].update(d_model=64, heads=4, layers=2, dropout=0.0,
                        pretrained_image_encoder=False, image_encoder="tiny")
    config["training"].update(batch_size=4, accumulation_steps=1,
                              epochs=args.tokenizer_epochs, tokenizer_warmup_epochs=100, validation_generation_limit=4,
                              tokenizer_lr=0.001, ar_lr=0.001)
    cfg = output/"config_tokenizer.json"
    cfg.write_text(json.dumps(config, indent=2))
    processed = output/"prepared"
    common = ["--device",args.device]
    call(["--stage","prepare","--config",cfg,"--sources",source,"--out_dir",processed],output)
    manifest = processed/"manifest.jsonl"
    call(["--stage","tokenizer","--config",cfg,"--manifest",manifest,"--out_dir",output/"tokenizer",*common],output)
    tokenizer = output/"tokenizer/best.pt"
    config["training"]["epochs"] = args.ar_epochs
    ar_cfg = output/"config_ar.json"
    ar_cfg.write_text(json.dumps(config,indent=2))
    call(["--stage","ar","--config",ar_cfg,"--manifest",manifest,"--tokenizer",tokenizer,"--out_dir",output/"ar",*common],output)
    for stage in ("tokenizer","ar"):
        call(["--stage","evaluate","--manifest",manifest,"--checkpoint",output/stage/"best.pt",
              "--tokenizer",tokenizer,"--out_dir",output/(stage+"_eval"),*common],output)
    record = next(json.loads(line) for line in manifest.read_text().splitlines() if json.loads(line)["split"]=="test")
    call(["--stage","predict","--checkpoint",output/"ar/best.pt","--tokenizer",tokenizer,
          "--input",processed/record["input_path"],"--out_dir",output/"prediction",*common],output)
    panels = figures(output)
    metrics = {stage:json.loads((output/(stage+"_eval")/"final_info.json").read_text()) for stage in ("tokenizer","ar")}
    summary = {"purpose":"Engineering demo; no accuracy or novelty claim",
               "dataset":"Poly Haven CC0, three panoramas; one scene per split",
               "torch":str(torch.__version__), "cuda_available":torch.cuda.is_available(),
               "metrics":metrics, "prediction":str(output/"prediction/prediction.hdr")}
    (output/"final_info.json").write_text(json.dumps(summary,indent=2))
    warning = "Engineering demo only: 3 real HDR panoramas, 1 training scene. Predictions are not a validated lighting estimator."
    (output/"notes.txt").write_text(warning+"\nTokenizer and AR trained separately; AR evaluation generates from BOS with no target tokens.\n")
    body = "".join(f'<h2>{html.escape(label)}</h2><img width="560" src="{name}">' for label,name in zip(
        ["Tokenizer reconstruction on held-out scene","Autoregressive generation on held-out scene"],panels))
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Lighting pipeline demo</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:16px sans-serif}aside{background:#fff1cc;padding:18px}img{max-width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>Single-image autoregressive HDR lighting demo</h1><aside>'+html.escape(warning)+
        '</aside>'+body+'<h2>Recorded results</h2><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>',encoding="utf-8")
    print("Demo complete:",output/"index.html",flush=True)


if __name__ == "__main__":
    main()
