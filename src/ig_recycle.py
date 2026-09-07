#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IG × Attention × GradNorm for model2
Supports train_mode / val_mode / both
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients

from model2 import InMemoryNPZDataset, collate_fn, BaseModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_npz", required=True)
    p.add_argument("--val_npz", required=True)
    p.add_argument("--label_csv", required=True)
    p.add_argument("--model_pth", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", default="both",
                   choices=["train", "val", "both"])
    p.add_argument("--topk", type=int, default=1000)
    p.add_argument("--embed_dim", type=int, default=1280)
    p.add_argument("--num_classes", type=int, default=11)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def save_checkpoint(idx, path):
    with open(path, "w") as f:
        json.dump({"idx": idx}, f)


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("idx", 0)
    return 0


def plot_heat(v, title, out_png):
    plt.figure(figsize=(12, 2))
    plt.imshow(v[np.newaxis, :], aspect="auto")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def build_model(args):
    model = BaseModel(
        embed_dim=args.embed_dim,
        num_classes=args.num_classes
    ).to(args.device)

    model.load_state_dict(
        torch.load(args.model_pth, map_location=args.device)
    )

    model.eval()

    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    return model


def forward_train(model, x, mask, importance):
    logits, _ = model(x, mask, importance)
    return logits


def forward_val(model, x, mask):
    logits, _ = model(x, mask, None)
    return logits


def grad_norm_train(model, seq, mask, importance, target):
    seq = seq.unsqueeze(0).detach().clone().requires_grad_(True)
    logits, _ = model(seq, mask.unsqueeze(0), importance.unsqueeze(0))
    logits[0, target].backward()
    return seq.grad.abs().mean(dim=-1).squeeze(0).detach()


def grad_norm_val(model, seq, mask, target):
    seq = seq.unsqueeze(0).detach().clone().requires_grad_(True)
    logits, _ = model(seq, mask.unsqueeze(0), None)
    logits[0, target].backward()
    return seq.grad.abs().mean(dim=-1).squeeze(0).detach()

def run_mode(args, dataset, split_name, explain_mode):
    outdir = os.path.join(
        args.output_dir,
        split_name
    )
    for s in ["ig_npz", "attn_npz", "fusion_npz", "plots"]:
        os.makedirs(os.path.join(outdir, s), exist_ok=True)

    #ckpt_file = os.path.join(outdir, "checkpoint.json")
    #start_idx = load_checkpoint(ckpt_file)
    start_idx = 0

    model = build_model(args)

    if explain_mode == "train":
        ig = IntegratedGradients(
            lambda x, mask, imp: forward_train(model, x, mask, imp)
        )
    else:
        ig = IntegratedGradients(
            lambda x, mask: forward_val(model, x, mask)
        )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    ranking = []
    counter = 0

    for batch in loader:

        x, importance, mask, y, names = batch

        if counter < start_idx:
            counter += 1
            continue

        x = x.to(args.device)
        importance = importance.to(args.device)
        mask = mask.to(args.device)

        seq = x[0]
        imp = importance[0]
        m = mask[0]
        name = names[0]

        with torch.no_grad():
            if explain_mode == "train":
                logits, attn = model(
                    seq.unsqueeze(0),
                    m.unsqueeze(0),
                    imp.unsqueeze(0)
                )
            else:
                logits, attn = model(
                    seq.unsqueeze(0),
                    m.unsqueeze(0),
                    None
                )
        print("attn shape =", attn.shape)



        pred = torch.sigmoid(logits[0]).argmax().item()

        if explain_mode == "train":

            ig_attr = ig.attribute(
                seq.unsqueeze(0),
                baselines=torch.zeros_like(seq.unsqueeze(0)),
                additional_forward_args=(
                    m.unsqueeze(0),
                    imp.unsqueeze(0)
                ),
                target=pred,
                n_steps=args.n_steps
            ).squeeze(0)
            grad = grad_norm_train(
                model,
                seq,
                m,
                imp,
                pred
            )

        else:

            ig_attr = ig.attribute(
                seq.unsqueeze(0),
                baselines=torch.zeros_like(seq.unsqueeze(0)),
                additional_forward_args=(
                    m.unsqueeze(0),
                ),
                target=pred,
                n_steps=args.n_steps
            ).squeeze(0)

            grad = grad_norm_val(
                model,
                seq,
                m,
                pred
            )
        attn = attn.squeeze(0)

        attn_score = attn.mean(dim=-1)

        attn_score = attn_score[1:]

        ig_score = ig_attr.abs().sum(dim=-1)



        assert len(attn_score) == len(ig_score)
        assert len(attn_score) == len(grad)

        fusion = (
                0.45 * zscore(ig_score)
                + 0.35 * zscore(attn_score)
                + 0.20 * zscore(grad)
        )

        fusion = fusion - fusion.min()

        fusion = fusion / (fusion.max() + 1e-8)

        # 防止后续pool出现0
        fusion = fusion + 1e-4

        # 再归一化
        fusion = fusion / fusion.sum()


        fusion_np = fusion.detach().cpu().numpy()

        top10 = np.argsort(fusion_np)[::-1][:10]

        np.savez_compressed(
            os.path.join(outdir, "attn_npz", f"{name}.npz"),
            attention=attn_score.detach().cpu().numpy(),
            pred=pred,
            mode=explain_mode,
        )
        np.savez_compressed(
            os.path.join(outdir, "fusion_npz", f"{name}.npz"),
            ig=ig_attr.detach().cpu().numpy(),
            ig_score=ig_score.detach().cpu().numpy(),
            attention=attn_score.detach().cpu().numpy(),
            mode=explain_mode,
            grad=grad.detach().cpu().numpy(),
            importance=imp.detach().cpu().numpy(),
            fusion=fusion_np,
            pred=pred,
            top10_idx=top10
        )

        plot_heat(
            fusion_np,
            f"{name}-{explain_mode}",
            os.path.join(outdir, "plots", f"{name}.png")
        )

        ranking.append((name, float(fusion_np.mean())))

        counter += 1

        #if counter % 50 == 0:
           # save_checkpoint(counter, ckpt_file)

    ranking = sorted(ranking, key=lambda x: x[1], reverse=True)[:args.topk]

    html = "<html><body><h1>TopK Fusion</h1>"
    for i, (name, score) in enumerate(ranking):
        html += f"<h3>{i+1}. {name}</h3>"
        html += f'<img src="plots/{name}.png" width="800">'
    html += "</body></html>"

    with open(os.path.join(outdir, "topk.html"), "w") as f:
        f.write(html)


def main():
    args = parse_args()

    train_ds = InMemoryNPZDataset(args.train_npz, args.label_csv)
    val_ds = InMemoryNPZDataset(args.val_npz, args.label_csv)

    modes = []
    if args.mode == "both":
        modes = ["train", "val"]
    else:
        modes = [args.mode]

    for mode in modes:
        run_mode(args, train_ds, "train", mode)
        run_mode(args, val_ds, "val", mode)

    print("Done.")


if __name__ == "__main__":
    main()