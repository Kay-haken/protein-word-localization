# =============================================
# IG × Attention × GradNorm × Class Motif Visual (Upgraded)
# =============================================

import os
import torch
import numpy as np
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
output_dir = "IG_30K_MOTIF_RESULTS_UPG"
topk_tokens = 20
topk_samples = 500
num_classes = 11

for split in ["train", "val"]:
    for sub in ["fusion_npz", "plots", "motifs_npz", "class_summary"]:
        os.makedirs(os.path.join(output_dir, split, sub), exist_ok=True)

# ----------------------------
# DATASET & DATALOADER
# ----------------------------
dataset = InMemoryNPZDataset(data_npz, label_csv, max_samples=30000)
train_idx = np.load("train_indices.npy")
val_idx = np.load("val_indices.npy")

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=False, collate_fn=collate_fn)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False, collate_fn=collate_fn)

# ----------------------------
# MODEL
# ----------------------------
model = BaseModel(embed_dim=1280, num_classes=num_classes, dropout_rate=0.15).to(device)
model.load_state_dict(torch.load("best_model_weights4.pth", map_location=device))
model.eval()

# ----------------------------
# IG / Attn / GradNorm
# ----------------------------
ig = IntegratedGradients(lambda x, l, m: model(x, l, m)[0])

def smooth_ig(seq, length, mask, target, n=10):
    total_attr = 0
    for _ in range(n):
        noise = torch.randn_like(seq) * 0.01
        attr = ig.attribute(
            seq + noise,
            baselines=torch.zeros_like(seq),
            additional_forward_args=(length, mask),
            target=target,
            n_steps=30
        )
        total_attr += attr
    return total_attr / n

def attention_rollout(attn):
    # attn: [layers, heads, seq, seq]
    attn = attn.mean(dim=1)  # [layers, seq, seq]
    I = torch.eye(attn.size(-1), device=attn.device)
    attn = attn + I
    attn = attn / attn.sum(dim=-1, keepdim=True)
    rollout = attn[0]
    for i in range(1, attn.size(0)):
        rollout = rollout @ attn[i]
    return rollout[0]  # CLS -> tokens

def grad_norm(seq, length, mask, target):
    seq = seq.unsqueeze(0).clone().detach().requires_grad_(True)
    logits, _ = model(seq, length.unsqueeze(0), mask.unsqueeze(0))
    logits[0, target].backward()
    grad = seq.grad.abs().sum(-1).squeeze(0)
    grad = grad / (grad.max() + 1e-8)
    return grad.detach()

def compute_fusion(seq, length, mask, target):
    ig_attr = smooth_ig(seq.unsqueeze(0), length.unsqueeze(0), mask.unsqueeze(0), target).squeeze(0)
    with torch.no_grad():
        _, attn = model(seq.unsqueeze(0), length.unsqueeze(0), mask.unsqueeze(0))
        attn = attn[0]
    attn_roll = attention_rollout(attn)
    grad = grad_norm(seq, length, mask, target)
    # probabilistic fusion
    ig_score = ig_attr.abs().sum(-1)
    ig_score = ig_score / (ig_score.sum() + 1e-8)
    attn_roll = attn_roll / (attn_roll.sum() + 1e-8)
    grad = grad / (grad.sum() + 1e-8)
    fusion = (ig_score ** 0.5) * (attn_roll ** 0.3) * (grad ** 0.2)
    return fusion.cpu().numpy()

# ----------------------------
# Motif 可视化
# ----------------------------
def plot_motif(class_id, topk_fusions, outdir):
    fusion_matrix = np.stack([
        np.pad(v, (0, max(0, topk_tokens - len(v))))[:topk_tokens]
        for v in topk_fusions
    ])
    plt.figure(figsize=(12, 6))
    sns.heatmap(fusion_matrix, cmap="Reds")
    plt.title(f"Class {class_id} TopK Motif Heatmap")
    plt.xlabel("Token Position")
    plt.ylabel("Top Samples")
    plt.tight_layout()
    plt.savefig(f"{outdir}/class{class_id}_motif.png", dpi=300)
    plt.close()
    np.savez_compressed(f"{outdir}/class{class_id}_motif.npz", fusion_matrix=fusion_matrix)

# ----------------------------
# RUN PIPELINE
# ----------------------------
def run_motif(loader, split):
    outdir = os.path.join(output_dir, split)
    class_bank = {i: [] for i in range(num_classes)}

    for batch in loader:
        x, lengths, mask, y, names, _ = batch
        x, lengths, mask = x.to(device), lengths.to(device), mask.to(device)
        for i in range(x.size(0)):
            seq = x[i]
            L = lengths[i]
            m = mask[i]
            name = names[i]
            with torch.no_grad():
                logits, _ = model(seq.unsqueeze(0), L.unsqueeze(0), m.unsqueeze(0))
                pred = logits[0].argmax().item()
            fusion = compute_fusion(seq, L, m, pred)
            class_bank[pred].append((fusion, name))
            np.savez_compressed(f"{outdir}/fusion_npz/{name}_class{pred}.npz", fusion=fusion, cls=pred, name=name)

    # class-wise consensus motif
    for cls, vals in class_bank.items():
        # prototype
        proto = np.mean([v[0] for v in vals], axis=0)
        # top samples by cosine similarity to prototype
        top_samples = sorted(vals, key=lambda x: np.dot(x[0], proto) / (np.linalg.norm(x[0])*np.linalg.norm(proto)+1e-8), reverse=True)[:topk_samples]
        topk_fusions = [v[0][:topk_tokens] for v in top_samples]
        plot_motif(cls, topk_fusions, f"{outdir}/motifs_npz")
        print(f"[{split}] class {cls} motif saved, topk_samples={len(top_samples)}")

# ----------------------------
# MAIN
# ----------------------------
for loader, split in [(train_loader, "train"), (val_loader, "val")]:
    print(f"\nRunning {split} motif pipeline...")
    run_motif(loader, split)
    print(f"{split} motif done")