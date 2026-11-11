"""Focused checks for gradients, causal masking, conditioning and generation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from src.lighting.models import AutoregressiveLighting, Tokenizer
from src.lighting.objectives import IlluminationObjective


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enhanced',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config=json.loads((ROOT/"configs/initial.json").read_text())
    config["data"].update(input_size=[32,32],env_size=[32,64])
    config["tokenizer"].update(base_channels=8,embedding_dim=8,codebook_size=16)
    config["ar"].update(image_encoder="tiny",pretrained_image_encoder=False,
                        d_model=32,heads=4,layers=1,dropout=0.0)
    if args.enhanced:
        config['tokenizer'].update(downsample_factor=8,residual_blocks=True)
        config['ar'].update(position_frequencies=[1,2,4,8],condition_prior=True)
        config['loss'].update(edge=.15,highlight=.5)
    tokenizer=Tokenizer(config)
    target=torch.rand(2,3,32,64)*4
    objective=IlluminationObjective(config)
    prediction,tokens,vq=tokenizer(torch.log1p(target))
    loss=objective(prediction,target)+vq
    loss.backward()
    assert torch.isfinite(loss) and tokenizer.codebook.weight.grad.abs().sum()>0
    assert tokenizer.encoder[0].conv.weight.grad.abs().sum()>0
    perfect=objective.metrics(torch.log1p(target),target)
    assert max(row["log_rmse"] for row in perfect)==0
    model=AutoregressiveLighting(config).eval()
    image=torch.rand(2,3,32,32)
    labels=tokens.flatten(1)
    changed=labels.clone()
    changed[:,3:]=(changed[:,3:]+1)%16
    with torch.no_grad():
        logits=model(image,labels)
        future=model(image,changed)
        torch.testing.assert_close(logits[:,:4],future[:,:4])
        swapped=model(1-image,labels)
        assert not torch.allclose(logits,swapped), "Image conditioning is disconnected"
        generated=model.generate(image)
        assert generated.shape==tokens.shape
        assert int(generated.min())>=0 and int(generated.max())<16
    model.train()
    model(image,labels).square().mean().backward()
    assert model.image_encoder[0].weight.grad.abs().sum()>0, "Random encoder is frozen"
    if args.enhanced:
        assert model.condition_queries.grad.abs().sum()>0
    print("PASS: VQ and image gradients, zero self-error, causal no-future-leakage, image conditioning, BOS-only generation. Enhanced:",args.enhanced)
    if torch.cuda.is_available():
        device=torch.device("cuda")
        tokenizer=tokenizer.to(device)
        prediction,_,vq=tokenizer(torch.log1p(target.to(device)))
        (prediction.square().mean()+vq).backward()
        torch.cuda.synchronize()
        print("PASS: CUDA tokenizer forward/backward on",torch.cuda.get_device_name())
    else:
        print("CPU verified; CUDA is unavailable in this environment.")


if __name__=="__main__":
    main()
