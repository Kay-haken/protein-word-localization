import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random
import argparse
from model1 import InMemoryNPZDataset, collate_fn, BaseModel,focal_loss,compute_pos_weight,AttentionHead




# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--npz_embeddings", type=str, required=True)
    parser.add_argument("--label_csv", type=str, required=True)

    # output
    parser.add_argument("--output_dir", type=str, required=True)

    # training hyperparams
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=800)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--weighted_decay", type=float, default=1e-4)
    parser.add_argument("--index_dir", type=str, default=None)

    return parser.parse_args()


# =========================
# Main
# =========================
def main():
    args = parse_args()

    # -------------------------
    # Seed & Device
    # -------------------------
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================
    # Dataset
    # =========================
    dataset = InMemoryNPZDataset(args.npz_embeddings, args.label_csv, max_samples=30000)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # save indices
    # =========================
    # index save dir
    # =========================
    index_dir = args.index_dir if args.index_dir is not None else args.output_dir
    os.makedirs(index_dir, exist_ok=True)

    np.save(
        os.path.join(index_dir, "train_indices.npy"),
        np.array(train_dataset.indices)
    )

    np.save(
        os.path.join(index_dir, "val_indices.npy"),
        np.array(val_dataset.indices)
    )

    print(f"Indices saved to: {index_dir}")
    print(f"Train size: {len(train_dataset.indices)} | Val size: {len(val_dataset.indices)}")

    # =========================
    # DataLoaders
    # =========================
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # =========================
    # Model / Optimizer / Scheduler
    # =========================
    model = BaseModel(embed_dim=1280, num_classes=11, dropout_rate=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weighted_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # =========================
    # Training
    # =========================
    best_val_acc = 0
    wait = 0
    train_loss_history, val_loss_history, all_metrics = [], [], []

    for epoch in range(args.max_epochs):
        print(f"\n===== Epoch {epoch+1}/{args.max_epochs} =====")
        model.train()
        epoch_train_loss = 0

        for x, lengths, mask, y, seq_names, ACCs in train_loader:
            x, lengths, mask, y = x.to(device), lengths.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            logits, attn = model(x, lengths, mask)
            loss = focal_loss(logits, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_loss = epoch_train_loss / len(train_loader)
        train_loss_history.append(train_loss)

        # =========================
        # Validation
        # =========================
        model.eval()
        epoch_val_loss = 0
        y_true_all, y_pred_all = [], []
        attn_all = []

        with torch.no_grad():
            for x, lengths, mask, y, seq_names, ACCs in val_loader:
                x, lengths, mask, y = x.to(device), lengths.to(device), mask.to(device), y.to(device)
                logits, attn = model(x, lengths, mask)
                loss = focal_loss(logits, y)
                epoch_val_loss += loss.item()

                for i, name in enumerate(seq_names):
                    attn_all.append({name: attn[i].cpu().numpy()})

                y_pred = torch.sigmoid(logits).cpu().numpy() > 0.5
                y_true_all.append(y.cpu().numpy())
                y_pred_all.append(y_pred)

        val_loss = epoch_val_loss / len(val_loader)
        val_loss_history.append(val_loss)

        # =========================
        # Metrics
        # =========================
        y_true_all = np.vstack(y_true_all)
        y_pred_all = np.vstack(y_pred_all)

        acc_total = accuracy_score(y_true_all, y_pred_all)
        precision_total = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        recall_total = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        f1_total = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        total_mcc = matthews_corrcoef(y_true_all.flatten(), y_pred_all.flatten())
        micro_f1 = f1_score(y_true_all.flatten(), y_pred_all.flatten(), zero_division=0)

        scheduler.step(acc_total)

        per_class_acc, per_class_mcc, per_class_f1 = {}, {}, {}
        for i in range(y_true_all.shape[1]):
            per_class_acc[f'class_{i}_acc'] = accuracy_score(y_true_all[:, i], y_pred_all[:, i])
            per_class_mcc[f'class_{i}_mcc'] = matthews_corrcoef(y_true_all[:, i], y_pred_all[:, i])
            per_class_f1[f'class_{i}_f1'] = f1_score(y_true_all[:, i], y_pred_all[:, i], zero_division=0)

        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': acc_total,
            'precision': precision_total,
            'recall': recall_total,
            'macro_f1': f1_total,
            'micro_f1': micro_f1,
            'total_mcc': total_mcc,
            **per_class_acc,
            **per_class_mcc,
            **per_class_f1
        }
        all_metrics.append(metrics)

        print(f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Accuracy={acc_total:.4f} | "
              f"Precision={precision_total:.4f} | Recall={recall_total:.4f} | Macro F1={f1_total:.4f} | Micro F1={micro_f1:.4f}")

        # =========================
        # Save best model + Early stopping
        # =========================
        if acc_total > best_val_acc:
            best_val_acc = acc_total
            wait = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_weights.pth"))
            torch.save(model, os.path.join(args.output_dir, "best_full_model.pth"))
            print(f"✔ Best model saved at epoch {epoch+1} | val_acc={best_val_acc:.4f}")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"⛔ Early stopping triggered at epoch {epoch+1}")
                break

    # =========================
    # Save metrics & attention
    # =========================
    pd.DataFrame(all_metrics).to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    pd.DataFrame(attn_all).to_csv(os.path.join(args.output_dir, "attention_values.csv"), index=False)
    print("✅ Training complete. Metrics and attention saved.")


if __name__ == "__main__":
    main()