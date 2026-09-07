
# =============================
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random
import os

# ----------------------------
# 固定随机种子

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

