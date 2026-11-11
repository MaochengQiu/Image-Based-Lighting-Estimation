"""Finish the condition-supervised run and save an honest before/after report."""
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_demo import figures
from src.lighting.models import AutoregressiveLighting,Tokenizer
from src.lighting.runtime import LightingDataset
from src.lighting.objectives import IlluminationObjective
from src.lighting.data import read_hdr

def main():
    os.environ["OMP_NUM_THREADS"]="2";os.environ["MKL_NUM_THREADS"]="2";torch.set_num_threads(2)
    out=ROOT/"runs/demo_optimized_v1"
    manifest=out/"prepared/manifest.jsonl";checkpoint=out/"ar_condition/best.pt";tok=out/"tokenizer/best.pt"
    for split in ("val","test"):
        directory=out/("ar_condition_val" if split=="val" else "ar_condition_eval")
        if not (directory/"final_info.json").exists():
            subprocess.run([sys.executable,str(ROOT/"experiment.py"),"--stage","evaluate","--manifest",str(manifest),
                "--checkpoint",str(checkpoint),"--tokenizer",str(tok),"--split",split,"--out_dir",str(directory),"--device","cpu"],cwd=ROOT,check=True)
    rows=[json.loads(x) for x in manifest.read_text().splitlines()];sample=next(x for x in rows if x["split"]=="test")
    directory=out/"prediction_condition"
    if not (directory/"inference.json").exists():
        subprocess.run([sys.executable,str(ROOT/"experiment.py"),"--stage","predict","--checkpoint",str(checkpoint),
            "--tokenizer",str(tok),"--input",str(out/"prepared"/sample["input_path"]),"--out_dir",str(directory),"--device","cpu"],cwd=ROOT,check=True)
    if not (out/"Figure_2_raw.png").exists():
        shutil.copyfile(out/"Figure_2.png",out/"Figure_2_raw.png")
    figures(out,ar_folder="ar_condition_eval")
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True);c=saved["config"]
    model=AutoregressiveLighting(c);model.load_state_dict(saved["model"]);model.eval()
    tokenizer=Tokenizer(c);tokenizer.load_state_dict(torch.load(tok,map_location="cpu",weights_only=True)["model"]);tokenizer.eval()
    val=LightingDataset(manifest,"val",c);x=torch.stack([val[i]["image"] for i in range(len(val))]);h=torch.stack([val[i]["hdr"] for i in range(len(val))])
    objective=IlluminationObjective(c)
    with torch.no_grad():
        generated=model.generate(x);shuffled=model.generate(x.flip(0))
        normal=objective.metrics(tokenizer.decode(generated),h);changed=objective.metrics(tokenizer.decode(shuffled),h)
    diagnostic={"changed_token_fraction":float((generated!=shuffled).float().mean()),
        "normal_log_rmse":sum(r["log_rmse"] for r in normal)/len(normal),
        "shuffled_log_rmse":sum(r["log_rmse"] for r in changed)/len(changed)}
    baseline=json.loads((ROOT/"runs/demo_20261110_warmup/final_info.json").read_text())
    result={"baseline":{n:baseline["metrics"][n]["metrics"] for n in ("tokenizer","ar")},
        "expanded_without_condition_loss":json.loads((out/"final_info.json").read_text())["metrics"],
        "optimized":{"tokenizer":json.loads((out/"tokenizer_eval/final_info.json").read_text())["metrics"],
                     "ar":json.loads((out/"ar_condition_eval/final_info.json").read_text())["metrics"]},
        "validation":{"baseline_ar":json.loads((out/"baseline_ar_val/final_info.json").read_text())["metrics"],
                      "optimized_ar":json.loads((out/"ar_condition_val/final_info.json").read_text())["metrics"]},
        "condition_check":diagnostic,"scope":"8 training scenes, 1 val and 1 test scene. No claim of benchmark generalization."}
    image=read_hdr(directory/"prediction.hdr");assert image.shape==(64,128,3) and np.isfinite(image).all()
    (out/"optimization_comparison.json").write_text(json.dumps(result,indent=2))
    body='<div class="pair"><section><h2>Previous demo</h2><img src="../demo_20261110_warmup/Figure_2.png"></section><section><h2>Optimized AR</h2><img src="Figure_2.png"></section></div>'
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Lighting optimization comparison</title>'
        '<style>body{max-width:1200px;margin:30px auto;font:16px sans-serif}.pair{display:flex;gap:20px}.pair section{width:49%}img{max-width:100%}pre{white-space:pre-wrap}aside{background:#fff1cc;padding:15px}</style>'
        '<h1>Autoregressive lighting: optimization results</h1><aside>Only 8 training scenes and 1 test scene. '
        'Tokenizer reconstruction and complete image-conditioned generation are different tasks. '
        'All previous failures are retained; changes combine additional data and architecture/loss modifications.</aside>'+body+
        '<h2>Optimized tokenizer reconstruction</h2><img width="560" src="Figure_1.png">'+
        '<h2>Recorded results</h2><pre>'+html.escape(json.dumps(result,indent=2))+'</pre>',encoding="utf-8")
    print(json.dumps(result,indent=2))
    print("Report:",out/"index.html")

if __name__=="__main__":
    main()
