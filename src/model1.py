import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# ----------------------------
def compute_pos_weight(labels):
    pos = labels.sum(0)
    neg = labels.shape[0] - pos
    pos_weight = neg / (pos + 1e-6)
    return pos_weight

def focal_loss(input, target, pos_weight=None, gamma=1.5):
    if pos_weight is not None:
        pos_weight = pos_weight.to(input.device)
    bce_loss = F.binary_cross_entropy_with_logits(
        input, target, pos_weight=pos_weight, reduction="none"
    )
    pt = torch.sigmoid(input) * target + (1 - torch.sigmoid(input)) * (1 - target)
    loss = ((1 - pt) ** gamma * bce_loss).mean()
    return loss

# ----------------------------
# 2. Attention Head
# ----------------------------
class AttentionHead(nn.Module):
    def __init__(self, hidden_dim, n_heads=8, dropout_rate=0.15):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.preattn_ln = nn.LayerNorm(hidden_dim // n_heads)
        self.Q = nn.Linear(hidden_dim // n_heads, n_heads, bias=False)
        torch.nn.init.normal_(self.Q.weight, mean=0.0, std=1 / (hidden_dim // n_heads))

    def forward(self, x, mask, lengths):
        B, L, D = x.shape
        x = x.view(B, L, self.n_heads, D // self.n_heads)
        x = self.preattn_ln(x)
        mul = (x * self.Q.weight.view(1, 1, self.n_heads, D // self.n_heads)).sum(-1)
        mul = mul.masked_fill(~mask.unsqueeze(-1), float("-1e9"))
        attn = F.softmax(mul, dim=1)
        attn = self.dropout(attn)
        x = (x * attn.unsqueeze(-1)).sum(1)
        x = x.view(B, -1)
        return x, attn.squeeze(-1)

# ----------------------------
# 3. Base Model
# ----------------------------
class BaseModel(nn.Module):
    def __init__(self, embed_dim, num_classes=11, dropout_rate=0.15):
        super().__init__()
        self.initial_ln = nn.LayerNorm(embed_dim)
        self.lin = nn.Linear(embed_dim, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_head = AttentionHead(512, n_heads=8, dropout_rate=dropout_rate)
        self.clf_head = nn.Linear(512, num_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x, lengths, mask):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), mask], dim=1)
        lengths = lengths + 1
        x = self.initial_ln(x)
        x = self.lin(x)
        x = F.relu(x)
        x = self.dropout(x)
        x_pool, x_attn = self.attn_head(x, mask, lengths)
        logits = self.clf_head(x_pool)
        return logits, x_attn

# ----------------------------
# 4. Dataset
# ----------------------------
class InMemoryNPZDataset(Dataset):
    def __init__(self, npz_file, label_csv, max_samples=None):
        self.data_npz = np.load(npz_file, allow_pickle=True)
        self.keys = list(self.data_npz.keys())
        if max_samples:
            self.keys = self.keys[:max_samples]

        self.label_df = pd.read_csv(label_csv).set_index('ACC')
        self.saved_labels = []

        self.embeddings, self.labels, self.masks, self.lengths = [], [], [], []

        label_cols = [
            'Membrane', 'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane',
            'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 'Lysosome/Vacuole',
            'Golgi apparatus', 'Peroxisome'
        ]

        for k in self.keys:
            emb = torch.tensor(self.data_npz[k], dtype=torch.float32)
            self.embeddings.append(emb)
            mask = torch.ones(len(emb), dtype=torch.bool)
            self.masks.append(mask)
            self.lengths.append(len(emb))

            seq_name_clean = re.sub(r'^\d+:\s*', '', str(k))
            lbl = torch.tensor(self.label_df.loc[seq_name_clean, label_cols].astype(float).values, dtype=torch.float32)
            self.labels.append(lbl)

            self.saved_labels.append({
                'sequence_name': seq_name_clean,
                'ACC': seq_name_clean,
                'labels': lbl.numpy()
            })

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.lengths[idx], self.masks[idx], self.labels[idx], self.keys[idx], self.keys[idx]

# ----------------------------
# 5. Collate_fn
# ----------------------------
def collate_fn(batch):
    embeddings, lengths, masks, labels, seq_names, ACCs = zip(*batch)
    embeddings_padded = pad_sequence(embeddings, batch_first=True)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    masks_tensor = pad_sequence(masks, batch_first=True)
    labels_tensor = torch.stack(labels, dim=0)
    return embeddings_padded, lengths_tensor, masks_tensor, labels_tensor, seq_names, ACCs

#