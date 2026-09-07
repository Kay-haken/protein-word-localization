# =========================================
# IG × Attention × GradNorm × Class Motif Visual (CLI Version)
# =========================================

import os
import json
import torch
import numpy as np
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
    parser = argparse.ArgumentParser(description="IG × Attention × GradNorm Pipeline")

    parser.add_argument("--npz_embeddings", type=str, required=True)
    parser.add_argument("--label_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_pth", type=str, required=True)
    parser.add_argument("--topk_tokens", type=int, default=20)
    parser.add_argument("--topk_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--index_dir", type=str, default=None)  # optional train/val indices

    return parser.parse_args()


# ----------------------------
# UTILITIES
# ----------------------------
def attention_rollout(attn):
    attn = attn.mean(dim=1)  # heads average
    I = torch.eye(attn.size(-1), device=attn.device)
    attn = attn + I
    attn = attn / attn.sum(dim=-1, keepdim=True)
    rollout = attn[0]
    for i in range(1, attn.size(0)):
        rollout = rollout @ attn[i]
    return rollout[0]

def grad_norm(model, seq, length, mask, target):
    seq = seq.unsqueeze(0).clone().detach().requires_grad_(True)
    logits, _ = model(seq, length.unsqueeze(0), mask.unsqueeze(0))
    logits[0, target].backward()
    return seq.grad.abs().sum(-1).squeeze(0).detach()

def compute_fusion(model, ig, seq, length, mask, target):
    ig_attr = ig.attribute(seq.unsqueeze(0),
                           additional_forward_args=(length.unsqueeze(0), mask.unsqueeze(0)),
                           target=target,
                           n_steps=30).squeeze(0)
    with torch.no_grad():
        _, attn = model(seq.unsqueeze(0), length.unsqueeze(0), mask.unsqueeze(0))
        attn = attn[0]
    attn_roll = attention_rollout(attn)
    grad = grad_norm(model, seq, length, mask, target)

    # probabilistic fusion
    ig_score = ig_attr.abs().sum(-1)
    ig_score = ig_score / (ig_score.sum() + 1e-8)
    attn_roll = attn_roll / (attn_roll.sum() + 1e-8)
    grad = grad / (grad.sum() + 1e-8)
    fusion = (ig_score ** 0.5) * (attn_roll ** 0.3) * (grad ** 0.2)
    return fusion.cpu().numpy()

def plot_motif(class_id, topk_fusions, outdir):
    fusion_matrix = np.stack([np.pad(v, (0, max(0, len(v) - len(v))))[:len(v)] for v in topk_fusions])
    plt.figure(figsize=(12, 6))
    sns.heatmap(fusion_matrix, cmap="Reds")
    plt.title(f"Class {class_id} TopK Motif Heatmap")
    plt.xlabel("Token Position")
    plt.ylabel("Top Samples")
    plt.tight_layout()
    plt.savefig(f"{outdir}/class{class_id}_motif.png", dpi=300)
    plt.close()
    np.savez_compressed(f"{outdir}/class{class_id}_motif.npz", fusion_matrix=fusion_matrix)

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"idx": 0}

def save_checkpoint(state, path):
    with open(path, "w") as f:
        json.dump(state, f)


# ----------------------------
# RUN PIPELINE
# ----------------------------
def run_pipeline(args, loader, split):
    outdir = os.path.join(args.output_dir, split)
    for sub in ["fusion_npz", "motifs_npz", "plots", "class_summary"]:
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)

    ckpt_path = os.path.join(outdir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt["idx"]

    device = args.device
    model = BaseModel(embed_dim=1280, num_classes=11, dropout_rate=0.15).to(device)
    model.load_state_dict(torch.load(args.model_pth, map_location=device))
    model.eval()

    ig = IntegratedGradients(lambda x, l, m: model(x, l, m)[0])

    class_bank = {i: [] for i in range(11)}
    counter = 0

    for batch in loader:
        x, lengths, mask, y, names, _ = batch
        x, lengths, mask = x.to(device), lengths.to(device), mask.to(device)

        for i in range(x.size(0)):
            if counter < start_idx:
                counter += 1
                continue

            seq = x[i]
            L = lengths[i]
            m = mask[i]
            name = names[i]

            with torch.no_grad():
                logits, _ = model(seq.unsqueeze(0), L.unsqueeze(0), m.unsqueeze(0))
                pred = logits[0].argmax().item()

            fusion = compute_fusion(model, ig, seq, L, m, pred)
            class_bank[pred].append((fusion, name))
            np.savez_compressed(f"{outdir}/fusion_npz/{name}_class{pred}.npz", fusion=fusion, cls=pred, name=name)

            counter += 1
            if counter % 50 == 0:
                save_checkpoint({"idx": counter}, ckpt_path)
                print(f"[{split}] checkpoint saved at {counter}")

    # class-wise motif
    for cls, vals in class_bank.items():
        if not vals:
            continue
        proto = np.mean([v[0] for v in vals], axis=0)
        top_samples = sorted(vals, key=lambda x: np.dot(x[0], proto) / (np.linalg.norm(x[0]) * np.linalg.norm(proto) + 1e-8),
                             reverse=True)[:args.topk_samples]
        topk_fusions = [v[0][:args.topk_tokens] for v in top_samples]
        plot_motif(cls, topk_fusions, os.path.join(outdir, "motifs_npz"))
        print(f"[{split}] class {cls} motif saved, topk_samples={len(top_samples)}")


# ----------------------------
# MAIN
# ----------------------------
def main():
    args = parse_args()
    dataset = InMemoryNPZDataset(args.npz_embeddings, args.label_csv, max_samples=30000)

    # indices
    if args.index_dir:
        train_idx = np.load(os.path.join(args.index_dir, "train_indices.npy"))
        val_idx = np.load(os.path.join(args.index_dir, "val_indices.npy"))
    else:
        n = len(dataset)
        split = int(0.8 * n)
        train_idx, val_idx = np.arange(split), np.arange(split, n)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    for loader, split in [(train_loader, "train"), (val_loader, "val")]:
        print(f"\nRunning {split} motif pipeline...")
        run_pipeline(args, loader, split)
        print(f"{split} done")


if __name__ == "__main__":
    main()