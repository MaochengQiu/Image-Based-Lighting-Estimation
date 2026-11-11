"""Render training and held-out comparisons for the completed AR experiments."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.lighting.models import AutoregressiveLighting,Tokenizer
from src.lighting.runtime import LightingDataset
from src.lighting.data import preview

OUT=ROOT/"runs/ar_controlled"
BASE=ROOT/"runs/demo_optimized_v1"


def tile(image,label):
    image=np.asarray(image)
    canvas=np.full((234,416,3),245,np.uint8)
    h,w=image.shape[:2]
    scale=min(384/w,192/h)
    resized=cv2.resize(image,(round(w*scale),round(h*scale)))
    value=np.round(np.clip(resized[...,::-1],0,1)*255).astype(np.uint8)
    canvas[34:34+value.shape[0],16:16+value.shape[1]]=value
    cv2.putText(canvas,label,(16,23),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1)
    return canvas


def save_panel(path,images):
    panels=[tile(image,label) for image,label in images]
    canvas=np.concatenate([np.concatenate(panels[:2],1),np.concatenate(panels[2:],1)],0)
    ok,data=cv2.imencode(".png",canvas)
    if not ok: raise ValueError("PNG encoding failed")
    data.tofile(path)


def read_rgb(path):
    image=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError(str(path))
    return image[...,::-1].astype(np.float32)/255


def main():
    torch.set_num_threads(2)
    saved=torch.load(OUT/"pretrained/last.pt",map_location="cpu",weights_only=True)
    config=saved["config"]
    dataset=LightingDataset(BASE/"prepared/manifest.jsonl","train",config)
    sample=dataset[0]
    tokenizer=Tokenizer(config)
    tokenizer.load_state_dict(torch.load(BASE/"tokenizer/best.pt",map_location="cpu",weights_only=True)["model"])
    tokenizer.eval()
    predictions=[]
    for checkpoint in (BASE/"ar_condition/last.pt",OUT/"pretrained/last.pt"):
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        model=AutoregressiveLighting(state["config"])
        model.load_state_dict(state["model"])
        model.eval()
        with torch.no_grad():
            log_hdr=tokenizer.decode(model.generate(sample["image"][None]))
            hdr=torch.expm1(log_hdr.clamp(0,20))[0].permute(1,2,0).numpy()
        predictions.append(preview(hdr))
    save_panel(OUT/"training_comparison.png",[
        (sample["image"].permute(1,2,0).numpy(),"TRAIN - Input"),
        (preview(sample["hdr"].permute(1,2,0).numpy()),"TRAIN - Ground truth"),
        (predictions[0],"TRAIN - Previous AR, last checkpoint"),
        (predictions[1],"TRAIN - Pretrained encoder, last checkpoint")])
    results=json.loads((OUT/"results.json").read_text())
    folder=OUT/"selected_test_pretrained"
    first=sorted(folder.glob("*_input.png"))[0]
    prefix=first.name[:-len("_input.png")]
    old=BASE/"ar_condition_eval"
    # Source indices differ across dataset preparations, use the same-scene optimized baseline.
    save_panel(OUT/"test_comparison.png",[
        (read_rgb(first),"TEST - Input"),
        (read_rgb(folder/(prefix+"_target.png")),"TEST - Ground truth"),
        (read_rgb(old/(prefix+"_predicted.png")),"TEST - Previous AR, validation-selected"),
        (read_rgb(folder/(prefix+"_predicted.png")),"TEST - Pretrained, validation-selected")])
    page=OUT/"index.html"
    text=page.read_text(encoding="utf-8")
    text=text.replace("<table>","<h2>Training example</h2><p>These last checkpoints demonstrate training-set fitting.</p><img width='832' src='training_comparison.png'><h2>Held-out scene</h2><p>These checkpoints were selected by validation. Test error did not improve.</p><img width='832' src='test_comparison.png'><h2>All runs</h2><table>",1)
    page.write_text(text,encoding="utf-8")
    print("Saved training and held-out comparison figures")


if __name__=="__main__":
    main()
