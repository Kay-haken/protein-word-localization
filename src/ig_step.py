# =========================================
# IG × Attention × GradNorm - 30k Stable Pipeline
# CLI Version
# =========================================

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from torch.utils.data import DataLoader, Subset
from captum.attr import IntegratedGradients

from model1 import InMemoryNPZDataset, collate_fn, BaseModel


# ----------------------------
# PARSE ARGS
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="IG × Attention × GradNorm 30k Pipeline")

    parser.add_argument("--npz_embeddings", type=str, required=True)
    parser.add_argument("--label_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_pth", type=str, required=True)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--index_dir", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


# ----------------------------
# UTILITIES
# ----------------------------
def attention_rollout(attn):
    attn = attn.mean(dim=-1)  # heads avg
    attn = attn[1:]  # remove CLS
    return attn / (attn.sum() + 1e-8)


def grad_norm(model, seq, length, mask, target):
    seq = seq.unsqueeze(0).clone().detach().requires_grad_(True)
    logits, _ = model(seq, length.unsqueeze(0), mask.unsqueeze(0))
    score = logits[0, target]
    score.backward()
    return seq.grad.norm(dim=-1).squeeze(0).detach()


def forward_for_ig(model, x, lengths, mask):
    logits, _ = model(x, lengths, mask)
    return logits


def compute(model, ig, seq, length, mask, target):
    ig_attr = ig.attribute(
        inputs=seq.unsqueeze(0),
        additional_forward_args=(length.unsqueeze(0), mask.unsqueeze(0)),
        target=target,
        n_steps=30
    ).squeeze(0).detach()

    with torch.no_grad():
        logits, attn = model(seq.unsqueeze(0), length.unsqueeze(0), mask.unsqueeze(0))
        attn = attn[0]

    attn_roll = attention_rollout(attn)
    grad = grad_norm(model, seq, length, mask, target)

    fusion = (ig_attr.abs().sum(-1) * attn_roll * grad).cpu().numpy()

    return ig_attr.cpu(), attn.cpu(), attn_roll.cpu(), grad.cpu(), fusion


def plot_fusion(name, fusion, outdir):
    plt.figure(figsize=(12, 2))
    sns.heatmap(fusion[np.newaxis, :], cmap="Reds")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"{outdir}/{name}.png", dpi=300)
    plt.close()


def save_checkpoint(state, path):
    with open(path, "w") as f:
        json.dump(state, f)


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"idx": 0}


# ----------------------------
# RUN PIPELINE
# ----------------------------
def run_pipeline(args, loader, split):
    outdir = os.path.join(args.output_dir, split)
    for sub in ["ig_npz", "attn_npz", "fusion_npz", "plots"]:
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)

    ckpt_path = os.path.join(outdir, "checkpoint_{cycle}.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt["idx"]

    all_results = []
    global_counter = 0

    device = args.device

    # MODEL
    model = BaseModel(embed_dim=1280, num_classes=11, dropout_rate=0.15).to(device)
    model.load_state_dict(torch.load(args.model_pth, map_location=device))
    model.eval()
    ig = IntegratedGradients(lambda x, lengths, mask: forward_for_ig(model, x, lengths, mask))

    for batch in loader:
        x, lengths, mask, y, names, _ = batch
        x, lengths, mask = x.to(device), lengths.to(device), mask.to(device)

        for i in range(x.size(0)):
            if global_counter < start_idx:
                global_counter += 1
                continue

            seq = x[i]
            L = lengths[i]
            m = mask[i]

            with torch.no_grad():
                logits, attn = model(seq.unsqueeze(0), L.unsqueeze(0), m.unsqueeze(0))
                pred = torch.sigmoid(logits[0]).argmax().item()

            ig_attr, attn_raw, attn_roll, grad, fusion = compute(model, ig, seq, L, m, pred)
            name = names[i]

            np.savez_compressed(f"{outdir}/ig_npz/{name}.npz", ig=ig_attr.numpy(), pred=pred)
            np.savez_compressed(f"{outdir}/attn_npz/{name}.npz", attn=attn_raw.numpy(), attn_roll=attn_roll.numpy(),
                                pred=pred)
            np.savez_compressed(f"{outdir}/fusion_npz/{name}.npz", fusion=fusion, pred=pred)
            plot_fusion(name, fusion, f"{outdir}/plots")

            all_results.append((name, fusion))
            global_counter += 1

            if global_counter % 50 == 0:
                save_checkpoint({"idx": global_counter}, ckpt_path)
                print(f"[{split}] checkpoint saved at {global_counter}")

    # TOPK
    top = sorted(all_results, key=lambda x: x[1].mean(), reverse=True)[:args.topk]
    html = "<html><body><h1>TopK IG×Attn×GradNorm</h1>"
    for i, (name, fusion) in enumerate(top):
        html += f"<h3>{i + 1}. {name}</h3>"
        html += f'<img src="plots/{name}.png" width="800">'
    html += "</body></html>"

    with open(f"{outdir}/topk.html", "w") as f:
        f.write(html)


# ----------------------------
# MAIN
# ----------------------------
def main():
    args = parse_args()
    dataset = InMemoryNPZDataset(args.npz_embeddings, args.label_csv, max_samples=30000)

    # load indices if index_dir provided
    if args.index_dir:
        train_idx = np.load(os.path.join(args.index_dir, "train_indices.npy"))
        val_idx = np.load(os.path.join(args.index_dir, "val_indices.npy"))
    else:
        # fallback: full dataset split
        n = len(dataset)
        split = int(0.8 * n)
        train_idx, val_idx = np.arange(split), np.arange(split, n)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=0)

    for loader, split in [(train_loader, "train"), (val_loader, "val")]:
        print(f"\nRunning {split} (30k safe mode)...")
        run_pipeline(args, loader, split)
        print(f"{split} done")


if __name__ == "__main__":
    main()