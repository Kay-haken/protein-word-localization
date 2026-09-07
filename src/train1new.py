# =============================
# Soft Top-K Subsequence Importance 加权 Loss (参数化版)
# =============================
import re
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef

# =========================================================
# 参数解析
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--val_npz", type=str, required=True)
    parser.add_argument("--label_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_out", type=str, default="recycle_model_weights.pth")

    parser.add_argument("--embed_dim", type=int, default=1280)
    parser.add_argument("--num_classes", type=int, default=11)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.7)

    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


# =========================================================
# 固定随机种子
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# ⭐ Focal Loss (multi-label)
# =========================================================
def focal_loss(input, target, gamma=1.5, pos_weight=None):
    if pos_weight is not None:
        pos_weight = pos_weight.to(input.device)
    bce = F.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight, reduction='none')
    pt = torch.sigmoid(input) * target + (1 - torch.sigmoid(input)) * (1 - target)
    loss = ((1 - pt) ** gamma) * bce
    return loss  # (B, C)


# =========================================================
# ⭐ Soft Top-K Importance
# =========================================================
def soft_topk_importance(importance, k=5, tau=0.7):
    importance = importance.clamp(min=0.0)
    B, L = importance.shape
    k = min(k, L)
    topk_vals, topk_idx = torch.topk(importance, k=k, dim=1)
    mask = torch.zeros_like(importance)
    mask.scatter_(1, topk_idx, 1.0)
    score = importance * mask
    soft_weights = torch.softmax(score / tau, dim=1)
    weighted_score = (importance * soft_weights).sum(dim=1)
    weighted_score = (weighted_score - weighted_score.mean()) / (weighted_score.std() + 1e-8)
    weighted_score = torch.sigmoid(weighted_score) * 2.0
    weighted_score = weighted_score.clamp(0.5, 2.0)
    return weighted_score  # (B,)


# =========================================================
# Attention Head & Base Model
# =========================================================
class AttentionHead(nn.Module):

    def __init__(
        self,
        hidden_dim,
        n_heads=8,
        dropout_rate=0.15
    ):
        super().__init__()

        self.n_heads = n_heads

        self.dropout = nn.Dropout(
            dropout_rate
        )

        self.preattn_ln = nn.LayerNorm(
            hidden_dim // n_heads
        )


        self.Q = nn.Linear(
            hidden_dim // n_heads,
            n_heads,
            bias=False
        )


        torch.nn.init.normal_(
            self.Q.weight,
            mean=0.0,
            std=1/(hidden_dim//n_heads)
        )



    def forward(
        self,
        x,
        mask,
        importance=None
    ):

        B,L,D = x.shape


        x = x.view(
            B,
            L,
            self.n_heads,
            D//self.n_heads
        )


        x = self.preattn_ln(x)



        # attention score
        attn = (
            x *
            self.Q.weight.view(
                1,
                1,
                self.n_heads,
                D//self.n_heads
            )
        ).sum(-1)



        # ==========================
        # importance bias
        # ==========================

        if importance is not None:

            importance = torch.log1p(
                importance
            )


            importance = (
                importance -
                importance.mean(dim=1,keepdim=True)
            ) / (
                importance.std(dim=1,keepdim=True)
                +1e-8
            )


            # scale
            importance = importance.unsqueeze(-1)


            attn = attn + 0.5 * importance



        # mask
        attn = attn.masked_fill(
            ~mask.unsqueeze(-1),
            -1e9
        )


        attn = F.softmax(
            attn,
            dim=1
        )


        attn = self.dropout(attn)



        x = (
            x *
            attn.unsqueeze(-1)
        ).sum(1)


        x=x.view(B,-1)


        return x,attn.squeeze(-1)


class BaseModel(nn.Module):

    def __init__(
        self,
        embed_dim=1280,
        num_classes=11,
        dropout_rate=0.15
    ):
        super().__init__()


        # 原来1280
        self.initial_ln = nn.LayerNorm(embed_dim)

        self.lin = nn.Linear(
            embed_dim,
            512
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.attn_head = AttentionHead(
            512,
            n_heads=8,
            dropout_rate=dropout_rate
        )


        self.clf_head = nn.Linear(
            512,
            num_classes
        )

    def forward(
            self,
            x,
            mask,
            importance=None
    ):
        B = x.size(0)

        cls_token = self.cls_token.expand(
            B, -1, -1
        )

        x = torch.cat(
            [
                cls_token,
                x
            ],
            dim=1
        )

        mask = torch.cat(
            [
                torch.ones(
                    B,
                    1,
                    dtype=torch.bool,
                    device=x.device
                ),
                mask
            ],
            dim=1
        )

        x = self.initial_ln(x)

        x = self.lin(x)

        x = F.relu(x)

        x = self.dropout(x)

        # 注意这里传 importance

        if importance is not None:
            importance = torch.cat(
                [
                    torch.zeros(
                        B,
                        1,
                        device=x.device
                    ),
                    importance
                ],
                dim=1
            )

        x_pool, attn = self.attn_head(
            x,
            mask,
            importance
        )

        logits = self.clf_head(
            x_pool
        )

        return logits, attn

# =========================================================
# Dataset & Collate
# =========================================================
class InMemoryNPZDataset(Dataset):
    def __init__(self, npz_file, label_csv, max_samples=None):
        self.data_npz = np.load(npz_file, allow_pickle=True)
        self.keys = list(self.data_npz.keys())
        if max_samples:
            self.keys = self.keys[:max_samples]

        self.label_df = pd.read_csv(label_csv).set_index('ACC')
        self.embeddings, self.labels, self.masks, self.importance_list = [], [], [], []

        label_cols = [
            'Membrane', 'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane',
            'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 'Lysosome/Vacuole',
            'Golgi apparatus', 'Peroxisome'
        ]
        self.indices = list(range(len(self.keys)))
        for k in self.keys:
            d = self.data_npz[k].item() if isinstance(self.data_npz[k], np.ndarray) else self.data_npz[k]
            emb = torch.tensor(d["embeddings"], dtype=torch.float32)
            imp = torch.tensor(d["importance"], dtype=torch.float32)
            self.embeddings.append(emb)
            self.importance_list.append(imp)
            mask = torch.ones(len(emb), dtype=torch.bool)
            self.masks.append(mask)

            seq_name_clean = re.sub(r'^\d+:\s*', '', str(k))
            lbl = torch.tensor(self.label_df.loc[seq_name_clean, label_cols].astype(float).values, dtype=torch.float32)
            self.labels.append(lbl)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.importance_list[idx], self.masks[idx], self.labels[idx], self.keys[idx]


def collate_fn(batch):
    embeddings, importance_list, masks, labels, seq_names = zip(*batch)
    embeddings_padded = pad_sequence(embeddings, batch_first=True)
    importance_padded = pad_sequence(importance_list, batch_first=True)
    labels_tensor = torch.stack(labels, dim=0)

    lengths = torch.tensor([len(m) for m in masks], dtype=torch.long)
    max_len = embeddings_padded.shape[1]
    masks_tensor = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return embeddings_padded, importance_padded, masks_tensor, labels_tensor, seq_names


# =========================================================
# 主训练函数
# =========================================================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset & Dataloader
    train_dataset = InMemoryNPZDataset(args.train_npz, args.label_csv)
    val_dataset   = InMemoryNPZDataset(args.val_npz, args.label_csv)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Model
    model = BaseModel(embed_dim=args.embed_dim, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0
    wait = 0
    train_loss_history, val_loss_history, all_metrics = [], [], []

    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")
        model.train()
        epoch_train_loss = 0.0

        for x, importance, mask, y, seq_names in train_loader:
            x, importance, mask, y = x.to(device), importance.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            logits, attn = model(x, mask, importance)
            loss_mat = focal_loss(
                logits,
                y
            )

            loss = torch.sqrt(
                loss_mat.sum(dim=1) + 1e-8
            ).mean()





            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        train_loss = epoch_train_loss / len(train_loader)
        train_loss_history.append(train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0
        y_true_all, y_pred_all = [], []

        with torch.no_grad():
            for x, importance, mask, y, seq_names in val_loader:
                x, importance, mask, y = x.to(device), importance.to(device), mask.to(device), y.to(device)

                logits, attn = model(
                    x,
                    mask,
                    None
                )
                loss = focal_loss(logits, y).mean()
                epoch_val_loss += loss.item()

                y_pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                y_true_all.append(y.cpu().numpy())
                y_pred_all.append(y_pred)

        val_loss = epoch_val_loss / len(val_loader)
        val_loss_history.append(val_loss)
        y_true_all = np.vstack(y_true_all)
        y_pred_all = np.vstack(y_pred_all)
        acc_total = accuracy_score(y_true_all, y_pred_all)
        recall_total = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        f1_total = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        scheduler.step(acc_total)

        per_class_acc, per_class_mcc, per_class_f1 = {}, {}, {}
        for i in range(y_true_all.shape[1]):
            per_class_acc[f'class_{i}_acc'] = accuracy_score(y_true_all[:, i], y_pred_all[:, i])
            per_class_mcc[f'class_{i}_mcc'] = matthews_corrcoef(y_true_all[:, i], y_pred_all[:, i])
            per_class_f1[f'class_{i}_f1'] = f1_score(y_true_all[:, i], y_pred_all[:, i], zero_division=0)

        total_mcc = matthews_corrcoef(y_true_all.flatten(), y_pred_all.flatten())
        micro_f1 = f1_score(y_true_all.flatten(), y_pred_all.flatten(), zero_division=0)
        metrics = {
            'epoch': epoch+1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': acc_total,
            'macro_f1': f1_total,
            'recall': recall_total,
            'micro_f1': micro_f1,
            'total_mcc': total_mcc,
            **per_class_acc,
            **per_class_mcc,
            **per_class_f1
        }
        all_metrics.append(metrics)
        print(f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Acc={acc_total:.4f} | F1={f1_total:.4f}")

        if acc_total > best_val_acc:
            best_val_acc = acc_total
            wait = 0
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.output_dir,
                    args.model_out
                )



            )
            print(f"✔ saved best model @ epoch {epoch+1}")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"⛔ early stopping @ epoch {epoch+1}")
                break

    # 保存日志
    pd.DataFrame(all_metrics).to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    print(f"✔ Metrics saved -> {os.path.join(args.output_dir, 'metrics.csv')}")


# =========================================================
# 入口
# =========================================================
if __name__ == "__main__":
    args = parse_args()
    main(args)