"""Apply and evaluate an explicit input-consistency step; keep raw AR outputs."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.lighting.data import read_hdr, write_rgb, preview, perspective, srgb
from src.lighting.objectives import IlluminationObjective
from src.lighting.visible import constrain_visible


def rgb(path):
    value=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
    if value is None: raise ValueError(str(path))
    return value[...,::-1].astype(np.float32)/255


def main():
    torch.set_num_threads(2)
    root=ROOT/"runs/demo_optimized_v1"
    config=json.loads((root/"config_condition.json").read_text())
    objective=IlluminationObjective(config)
    rows=[json.loads(x) for x in (root/"prepared/manifest.jsonl").read_text().splitlines()]
    folder=root/"visible_consistent";folder.mkdir(exist_ok=True)
    report={"method":"Raw autoregressive prediction + geometrically projected observed LDR. No target is used for projection or scale estimation.",
            "exposure_policy":"Estimated from raw prediction, not read from target or capture metadata.",
            "metrics":{}}
    def tensor(x): return torch.from_numpy(x.transpose(2,0,1).copy())[None]
    for split in ("val","test"):
        results=[]
        source=root/("ar_condition_val" if split=="val" else "ar_condition_eval")
        samples=[x for x in rows if x["split"]==split]
        for row in samples:
            prediction=read_hdr(source/(row["sample_id"]+"_predicted.hdr"))
            image=rgb(root/"prepared"/row["input_path"])
            refined,info=constrain_visible(prediction,image,config["data"]["hfov_degrees"])
            target=read_hdr(root/"prepared"/row["target_path"])
            metrics=objective.metrics(torch.log1p(tensor(refined)/objective.scale),tensor(target))[0]
            results.append({"sample_id":row["sample_id"],**metrics,**info})
            write_rgb(folder/(row["sample_id"]+".hdr"),refined,hdr=True)
            write_rgb(folder/(row["sample_id"]+".png"),preview(refined))
            if row==samples[0] and split=="test":
                strips=[]
                for title,im in [("Input photo",image),("Ground truth",preview(target)),
                                  ("Raw autoregressive",preview(prediction)),("AR + observed view",preview(refined))]:
                    canvas=np.full((200,560,3),245,np.uint8)
                    h,w=im.shape[:2];ratio=min(512/w,144/h)
                    im=cv2.resize(im,(round(w*ratio),round(h*ratio)))
                    canvas[40:40+im.shape[0],16:16+im.shape[1]]=np.round(im[...,::-1]*255).astype(np.uint8)
                    cv2.putText(canvas,title,(16,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1)
                    strips.append(canvas)
                ok,data=cv2.imencode(".png",np.concatenate(strips))
                if not ok: raise ValueError("PNG encode failed")
                data.tofile(root/"visible_comparison.png")
        keys=("log_rmse","linear_rmse","diffuse_rmse","energy_relative_error")
        report["metrics"][split]={key:sum(r[key] for r in results)/len(results) for key in keys}
        report["metrics"][split]["samples"]=len(results)
        (folder/(split+".json")).write_text(json.dumps(results,indent=2))
    # Geometric contract: a known-exposure constant source must be preserved in the
    # unobserved half, and recovered in the centre of the observed region.
    predicted=np.full((64,128,3),0.2,np.float32)
    photograph=srgb(np.full((64,64,3),0.5,np.float32))
    result,_=constrain_visible(predicted,photograph,90,exposure=1)
    np.testing.assert_allclose(result[32,64],0.5,atol=.005)
    np.testing.assert_array_equal(result[:,:32],predicted[:,:32])
    np.testing.assert_array_equal(result[:,96:],predicted[:,96:])
    (root/"visible_consistency.json").write_text(json.dumps(report,indent=2))
    page=root/"index.html";text=page.read_text(encoding="utf-8")
    text=text.replace("<h2>Recorded results</h2>","<h2>Explicit observed-view consistency (not learned recovery)</h2><p>The view outside the camera remains the raw autoregressive result. Saturated pixels only give lower bounds. Scale is estimated without ground truth.</p><img width='560' src='visible_comparison.png'><h2>Recorded results</h2>")
    page.write_text(text,encoding="utf-8")
    print(json.dumps(report,indent=2))
    print("PASS: projection centre and unchanged unobserved hemisphere")


if __name__=="__main__":
    main()
