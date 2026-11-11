"""Expanded real-HDR demo optimization; validation chooses checkpoints, never test."""
import argparse
import csv
import hashlib
import html
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_demo import figures

TRAIN = ["small_empty_room_1","entrance_hall","wooden_lounge","artist_workshop",
         "modern_bathroom","small_empty_house","country_club","photo_studio_01"]
ASSETS = [(x,"train") for x in TRAIN] + [("lebombo","val"),("studio_small_08","test")]


def fetch(url):
    req=urllib.request.Request(url,headers={"User-Agent":"LightingEstimationResearchDemo/0.1"})
    with urllib.request.urlopen(req,timeout=60) as response:
        return response.read()


def download(item):
    asset,split=item
    raw=ROOT/"data/raw/polyhaven_demo"
    target=raw/(asset+".hdr")
    old=json.loads((ROOT/"data/manifests/polyhaven_demo/provenance.json").read_text())
    cached=next((a for a in old["assets"] if a["asset"]==asset),None)
    if cached and target.exists() and hashlib.md5(target.read_bytes()).hexdigest()==cached["md5"]:
        return {**cached,"split":split,"path":str(target)}
    for attempt in range(3):
        try:
            info=json.loads(fetch("https://api.polyhaven.com/files/"+asset))["hdri"]["1k"]["hdr"]
            payload=target.read_bytes() if target.exists() else fetch(info["url"])
            if len(payload)!=info["size"] or hashlib.md5(payload).hexdigest()!=info["md5"]:
                raise ValueError("Checksum mismatch for "+asset)
            if not target.exists():
                target.write_bytes(payload)
            return dict(asset=asset,split=split,path=str(target),url=info["url"],
                        md5=info["md5"],sha256=hashlib.sha256(payload).hexdigest(),license="CC0")
        except Exception:
            if attempt==2: raise
            time.sleep(1+attempt)


def main():
    os.environ["OMP_NUM_THREADS"]="2"
    os.environ["MKL_NUM_THREADS"]="2"
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'runs/demo_optimized_v2')
    args=parser.parse_args()
    out=args.out_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/"final_info.json").exists():
        raise ValueError("Completed run exists; preserve it and choose a new run directory")
    provenance=out/"downloaded.json"
    if not provenance.exists():
        records=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(download,x) for x in ASSETS]):
                row=future.result();records.append(row);print("HDR ready:",row["asset"],flush=True)
        provenance.write_text(json.dumps(sorted(records,key=lambda r:r["asset"]),indent=2))
    records=json.loads(provenance.read_text())
    source=out/"sources.csv"
    with source.open("w",encoding="utf-8",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=["scene_id","pano_path","split"]);w.writeheader()
        w.writerows(dict(scene_id=r["asset"],pano_path=r["path"],split=r["split"]) for r in records)
    config=json.loads((ROOT/"runs/demo_20261110_warmup/config_tokenizer.json").read_text())
    config["tokenizer"].update(base_channels=24,embedding_dim=32,codebook_size=128,
                                downsample_factor=8,residual_blocks=True)
    config["ar"].update(d_model=96,layers=3,heads=4,dropout=0.1,position_frequencies=[1,2,4,8],condition_prior=True,condition_loss_weight=1.0)
    config["loss"].update(linear=0.2,edge=0.15,highlight=0.5)
    config["training"].update(batch_size=8,epochs=100,tokenizer_warmup_epochs=80,
                               tokenizer_lr=0.0005,ar_lr=0.0003,validation_interval=10)
    cfg=out/"config_tokenizer.json";cfg.write_text(json.dumps(config,indent=2))
    def call(stage,*args):
        log=out/(stage+".log")
        print("Starting",stage,"log:",log,flush=True)
        with log.open("w",encoding="utf-8") as stream:
            subprocess.run([sys.executable,"-u",str(ROOT/"experiment.py"),*map(str,args)],
                           cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print("Finished",stage,flush=True)
    processed=out/"prepared"
    if not (processed/"manifest.jsonl").exists():
        call("prepare","--stage","prepare","--config",cfg,"--sources",source,"--out_dir",processed)
    manifest=processed/"manifest.jsonl";tok=out/"tokenizer/best.pt"
    if not tok.exists():
        call("tokenizer","--stage","tokenizer","--config",cfg,"--manifest",manifest,"--out_dir",out/"tokenizer","--device","cpu")
    config["training"]["epochs"]=100
    ar_cfg=out/"config_ar.json";ar_cfg.write_text(json.dumps(config,indent=2))
    if not (out/"ar/best.pt").exists():
        call("ar","--stage","ar","--config",ar_cfg,"--manifest",manifest,"--tokenizer",tok,"--out_dir",out/"ar","--device","cpu")
    # Validation is inspected first; test is a final descriptive comparison only.
    for name in ("tokenizer","ar"):
        for split in ("val","test"):
            folder=out/(name+"_val" if split=="val" else name+"_eval")
            if not (folder/"final_info.json").exists():
                call(name+"_"+split,"--stage","evaluate","--manifest",manifest,"--checkpoint",out/name/"best.pt",
                     "--tokenizer",tok,"--split",split,"--out_dir",folder,"--device","cpu")
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    sample=next(r for r in rows if r["split"]=="test")
    if not (out/"prediction/inference.json").exists():
        call("predict","--stage","predict","--checkpoint",out/"ar/best.pt","--tokenizer",tok,
             "--input",processed/sample["input_path"],"--out_dir",out/"prediction","--device","cpu")
    figures(out)
    import torch
    from src.lighting.models import AutoregressiveLighting
    from src.lighting.runtime import LightingDataset
    torch.set_num_threads(2)
    saved=torch.load(out/"ar/best.pt",map_location="cpu",weights_only=True)
    model=AutoregressiveLighting(saved["config"]);model.load_state_dict(saved["model"]);model.eval()
    val=LightingDataset(manifest,"val",saved["config"])
    images=torch.stack([val[i]["image"] for i in range(len(val))])
    tokens=model.generate(images)
    sensitivity=float((tokens!=model.generate(images.flip(0))).float().mean())
    result={"purpose":"Small real-data demo; not a benchmark",
            "training_scenes":len(TRAIN),"val_scene":"lebombo","test_scene":"studio_small_08",
            "changed_token_fraction_when_shuffling_val_images":sensitivity,
            "metrics":{name:json.loads((out/(name+"_eval")/"final_info.json").read_text())["metrics"] for name in ("tokenizer","ar")},
            "validation":{name:json.loads((out/(name+"_val")/"final_info.json").read_text())["metrics"] for name in ("tokenizer","ar")}}
    (out/"final_info.json").write_text(json.dumps(result,indent=2))
    # Same test scenes and views as previous demo. This compares a package of changes,
    # including more training data; it is not an architecture ablation.
    baseline=json.loads((ROOT/"runs/demo_20261110_warmup/final_info.json").read_text())
    comparison={"baseline":{n:baseline["metrics"][n]["metrics"] for n in ("tokenizer","ar")},"optimized":result["metrics"]}
    (out/"comparison.json").write_text(json.dumps(comparison,indent=2))
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Lighting optimization</title>'
        '<style>body{max-width:1200px;margin:30px auto;font:16px sans-serif}.pair{display:flex;gap:20px}.pair img{width:48%;object-fit:contain;align-self:flex-start}pre{white-space:pre-wrap}</style>'
        '<h1>Autoregressive lighting demo: before and after</h1><p>Real HDR, eight training scenes; CPU. '
        'Same validation/test scenes as baseline. Small demonstration only. Additional data and model changes are combined, not isolated contributions.</p>'
        '<div class="pair"><img src="../demo_20261110_warmup/Figure_2.png"><img src="Figure_2.png"></div>'
        '<h2>Tokenizer reconstruction</h2><img width="560" src="Figure_1.png">'
        '<h2>Recorded comparison</h2><pre>'+html.escape(json.dumps(comparison,indent=2))+'</pre>',encoding="utf-8")
    (out/"notes.txt").write_text("Expanded training data and model package; not a controlled architecture ablation.\n"+json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    print("Report:",out/"index.html",flush=True)


if __name__=="__main__":
    main()
