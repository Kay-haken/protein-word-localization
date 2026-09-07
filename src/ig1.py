# =========================================
# IG × Attention × GradNorm - 30k Stable Pipeline
# Resume + TopK + Full Save Version
# =========================================

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from captum.attr import IntegratedGradients

from model1 import InMemoryNPZDataset, collate_fn, BaseModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# CONFIG
# ----------------------------
data_npz = "/path/to/project/sequence_embeddings_avg_pool3.npz"
label_csv = "/path/to/project/multisub_5_partitions_unique.csv"

output_dir = "IG_30K_RESULTS"
topk = 1000

for split in ["train", "val"]:
    for sub in ["ig_npz", "attn_npz", "fusion_npz", "plots"]:
        os.makedirs(os.path.join(output_dir, split, sub), exist_ok=True)

os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# DATASET (30k)
# ----------------------------
dataset = InMemoryNPZDataset(data_npz, label_csv, max_samples=30000)

train_idx = np.load("train_indices.npy")
val_idx = np.load("val_indices.npy")

train_loader = DataLoader(
    Subset(dataset, train_idx),
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

val_loader = DataLoader(
    Subset(dataset, val_idx),
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

# ----------------------------
# MODEL
# ----------------------------
model = BaseModel(embed_dim=1280, num_classes=11, dropout_rate=0.15).to(device)
model.load_state_dict(torch.load("best_model_weights4.pth", map_location=device))
model.eval()

# ----------------------------
# IG
# ----------------------------
def forward_for_ig(x, lengths, mask):
    logits, _ = model(x, lengths, mask)
    return logits

ig = IntegratedGradients(forward_for_ig)

# ----------------------------
# Attention rollout
# ----------------------------
def attention_rollout(attn):
    attn = attn.mean(dim=-1)  # heads avg
    attn = attn[1:]           # remove CLS
    return attn / (attn.sum() + 1e-8)

# ----------------------------
# GradNorm
# ----------------------------
def grad_norm(seq, length, mask, target):
    seq = seq.unsqueeze(0).clone().detach().requires_grad_(True)
    logits, _ = model(seq, length.unsqueeze(0), mask.unsqueeze(0))
    score = logits[0, target]
    score.backward()
    return seq.grad.norm(dim=-1).squeeze(0).detach()

# ----------------------------
# COMPUTE SINGLE SAMPLE
# ----------------------------
def compute(seq, length, mask, target):
    ig_attr = ig.attribute(
        inputs=seq.unsqueeze(0),
        additional_forward_args=(length.unsqueeze(0), mask.unsqueeze(0)),
        target=target,
        n_steps=30
    ).squeeze(0).detach()

    with torch.no_grad():
        logits, attn = model(seq.unsqueeze(0), length.squeeze(0), mask.unsqueeze(0))
        attn = attn[0]

    attn_roll = attention_rollout(attn)
    grad = grad_norm(seq, length, mask, target)

    fusion = (ig_attr.abs().sum(-1) * attn_roll * grad).cpu().numpy()

    return ig_attr.cpu(), attn.cpu(), attn_roll.cpu(), grad.cpu(), fusion

# ----------------------------
# PLOT
# ----------------------------
def plot_fusion(name, fusion, outdir):
    plt.figure(figsize=(12, 2))
    sns.heatmap(fusion[np.newaxis, :], cmap="Reds")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"{outdir}/{name}.png", dpi=300)
    plt.close()

# ----------------------------
# SAVE CHECKPOINT (VERY IMPORTANT FOR 30K)
# ----------------------------
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
def run(loader, split):
    outdir = os.path.join(output_dir, split)
    ckpt_path = os.path.join(outdir, "checkpoint.json")

    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt["idx"]

    all_results = []
    global_counter = 0

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

            ig_attr, attn_raw, attn_roll, grad, fusion = compute(seq, L, m, pred)

            name = names[i]

            # SAVE EACH
            np.savez_compressed(f"{outdir}/ig_npz/{name}.npz",
                                ig=ig_attr.numpy(),
                                pred=pred)

            np.savez_compressed(f"{outdir}/attn_npz/{name}.npz",
                                attn=attn_raw.numpy(),
                                attn_roll=attn_roll.numpy(),
                                pred=pred)

            np.savez_compressed(f"{outdir}/fusion_npz/{name}.npz",
                                fusion=fusion,
                                pred=pred)

            plot_fusion(name, fusion, f"{outdir}/plots")

            all_results.append((name, fusion))

            global_counter += 1

            # SAVE CHECKPOINT EVERY 50 SAMPLES
            if global_counter % 50 == 0:
                save_checkpoint({"idx": global_counter}, ckpt_path)
                print(f"[{split}] checkpoint saved at {global_counter}")

    # ---------------- TOPK ----------------
    top = sorted(all_results, key=lambda x: x[1].mean(), reverse=True)[:topk]

    html = "<html><body><h1>TopK IG×Attn×GradNorm</h1>"
    for i, (name, fusion) in enumerate(top):
        html += f"<h3>{i+1}. {name}</h3>"
        html += f'<img src="plots/{name}.png" width="800">'
    html += "</body></html>"

    with open(f"{outdir}/topk.html", "w") as f:
        f.write(html)

# ----------------------------
# MAIN
# ----------------------------
for loader, split in [(train_loader, "train"), (val_loader, "val")]:
    print(f"\nRunning {split} (30k safe mode)...")
    run(loader, split)
    print(f"{split} done")