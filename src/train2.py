# =============================
# Soft Top-K Subsequence Importance + Multi-Query Perceiver (升级完整版)
# =============================
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random
import os

# ----------------------------
# 固定随机种子
# ----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------------
# ⭐ Focal Loss (multi-label)
# ----------------------------
def focal_loss(input, target, gamma=1.5, pos_weight=None):
    if pos_weight is not None:
        pos_weight = pos_weight.to(input.device)
    bce = F.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight, reduction='none')
    pt = torch.sigmoid(input) * target + (1 - torch.sigmoid(input)) * (1 - target)
    loss = ((1 - pt) ** gamma) * bce
    return loss  # (B, C)

# ----------------------------
# ⭐ Soft Top-K Importance
# ----------------------------
def soft_topk_importance_v2(importance, k=5, tau=0.7):
    importance = importance.clamp(min=0.0)
    B, L = importance.shape

    k = min(k, L)
    topk_idx = torch.topk(importance, k=k, dim=1).indices

    mask = torch.zeros_like(importance)
    mask.scatter_(1, topk_idx, 1.0)

    imp = importance * mask
    imp = imp / (imp.sum(dim=1, keepdim=True) + 1e-8)
    return imp.sum(dim=1)   # (B,)

# ----------------------------
# 🔥 Perceiver-style Multi-Query Attention
# ----------------------------
class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim=512, num_labels=11, n_heads=8, dim_feedforward=1024, dropout=0.1, n_query=4):
        super().__init__()
        self.num_labels = num_labels
        self.n_query = n_query
        self.embed_dim = embed_dim

        # 每个 label 对应一个 query token
        self.query = nn.Parameter(torch.randn(num_labels, n_query, embed_dim))  # (num_labels, n_query, D)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ),
            num_layers=2
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, importance=None):
        """
        x: (B, L, D)
        mask: (B, L)
        importance: (B, L)
        """
        B, L, D = x.shape

        # 🔥 replicate query tokens per batch
        queries = self.query.unsqueeze(0).expand(B, -1, -1, -1)  # (B, num_labels, n_query, D)
        queries = queries.reshape(B, self.num_labels * self.n_query, D)  # (B, num_labels*n_query, D)

        # Concatenate queries + sequence
        x_all = torch.cat([queries, x], dim=1)  # (B, Q+L, D)

        # mask
        query_mask = torch.ones(B, self.num_labels * self.n_query, dtype=torch.bool, device=x.device)
        mask_all = torch.cat([query_mask, mask], dim=1)  # (B, Q+L)

        # ---- Transformer Encoder ----
        x_all = self.encoder(x_all, src_key_padding_mask=~mask_all)

        # 🔥 importance gating
        if importance is not None:
            imp = importance.unsqueeze(1)  # (B,1,L)
            imp = imp / (imp.mean(dim=-1, keepdim=True)+1e-8)
            # gate sequence tokens
            seq_x = x_all[:, self.num_labels * self.n_query:]  # (B, L, 512)

            imp = importance.unsqueeze(-1)  # (B, L, 1)

            imp = imp / (imp.mean(dim=1, keepdim=True) + 1e-8)

            gate = 1 + 0.5 * (imp - imp.mean(dim=1, keepdim=True))

            x_all[:, self.num_labels * self.n_query:] = seq_x * gate

        x_all = self.dropout(x_all)

        # Perceiver-style pooling: 每个 label 对应 n_query tokens 平均
        pooled = []
        for i in range(self.num_labels):
            q_slice = x_all[:, i*self.n_query:(i+1)*self.n_query]
            pooled.append(q_slice.mean(dim=1))
        x_pool = torch.stack(pooled, dim=1)  # (B, num_labels, D)
        x_pool = x_pool.mean(dim=1)  # 最终 pooling -> (B, D)

        return x_pool, None

# ----------------------------
# Base Model
# ----------------------------
class BaseModel(nn.Module):
    def __init__(self, embed_dim, num_classes=11, dropout_rate=0.15):
        super().__init__()
        self.initial_ln = nn.LayerNorm(embed_dim)
        self.lin = nn.Linear(embed_dim, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_head = MultiQueryAttention(embed_dim=512, num_labels=num_classes, n_heads=8, dim_feedforward=1024, dropout=dropout_rate)
        self.clf_head = nn.Linear(512, num_classes)

    def forward(self, x, mask, importance=None):
        B = x.size(0)
        x = self.initial_ln(x)
        x = self.lin(x)
        x = F.relu(x)
        x = self.dropout(x)
        x_pool, x_attn = self.attn_head(x, mask, importance)
        logits = self.clf_head(x_pool)
        return logits, x_attn

# ----------------------------
# Dataset
# ----------------------------
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
        self.indices = list(range(len(self.keys)))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.importance_list[idx], self.masks[idx], self.labels[idx], self.keys[idx]

# ----------------------------
# Collate Function
# ----------------------------
def collate_fn(batch):
    embeddings, importance_list, masks, labels, seq_names = zip(*batch)
    embeddings_padded = pad_sequence(embeddings, batch_first=True)
    importance_padded = pad_sequence(importance_list, batch_first=True)
    labels_tensor = torch.stack(labels, dim=0)

    # 正确 mask
    lengths = torch.tensor([len(m) for m in masks], dtype=torch.long)
    max_len = embeddings_padded.shape[1]
    masks_tensor = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return embeddings_padded, importance_padded, masks_tensor, labels_tensor, seq_names

# ----------------------------
# 配置 & 数据
#-------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
label_csv = "/path/to/project/multisub_5_partitions_unique.csv"

train_npz = "/path/to/project/sequence_embeddings_avg_pool_subseqs_with_importance.npz"
val_npz   = "/path/to/project/sequence_embeddings_avg_pool_subseqs_with_importance_val.npz"

train_dataset = InMemoryNPZDataset(train_npz, label_csv, max_samples=30000)
val_dataset   = InMemoryNPZDataset(val_npz, label_csv, max_samples=30000)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
train_indices = train_dataset.indices
val_indices = val_dataset.indices
np.save("train_indices_v3.npy", np.array(train_indices))
np.save("val_indices_v3.npy", np.array(val_indices))
print(f"Train size: {len(train_indices)} saved -> train_indices.npy")
print(f"Val size: {len(val_indices)} saved -> val_indices.npy")
model = BaseModel(embed_dim=1280, num_classes=11, dropout_rate=0.15).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

n_epochs = 200
patience = 20
wait = 0
best_val_acc = 0
train_loss_history, val_loss_history, all_metrics = [], [], []

# ----------------------------
# 训练循环
# ----------------------------
for epoch in range(n_epochs):
    print(f"\n===== Epoch {epoch+1}/{n_epochs} =====")
    model.train()
    epoch_train_loss = 0.0

    for x, importance, mask, y, seq_names in train_loader:
        x, importance, mask, y = x.to(device), importance.to(device), mask.to(device), y.to(device)

        optimizer.zero_grad()
        logits, attn = model(x, mask, importance)



        loss_mat = focal_loss(logits, y)
        loss_per_sample = torch.sqrt(loss_mat.sum(dim=1) + 1e-8)

        loss = focal_loss(logits, y).mean() # ✅ 改成 mean

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_train_loss += loss.item()

    train_loss = epoch_train_loss / len(train_loader)
    train_loss_history.append(train_loss)

    # ----------------------------
    # Validation
    # ----------------------------
    model.eval()
    epoch_val_loss = 0
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for x, importance, mask, y, seq_names in val_loader:
            x, importance, mask, y = x.to(device), importance.to(device), mask.to(device), y.to(device)

            logits, attn = model(x, mask, importance=None)
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

    # ----------------------------
    # Early Stopping
    # ----------------------------
    if acc_total > best_val_acc:
        best_val_acc = acc_total
        wait = 0
        torch.save(model.state_dict(), "best_model_softtopk_v3.pth")
        print(f"✔ saved best model @ epoch {epoch+1}")
    else:
        wait += 1
        if wait >= patience:
            print(f"⛔ early stopping @ epoch {epoch+1}")
            break

# ----------------------------
# 保存日志 & 绘图
# ----------------------------
pd.DataFrame(all_metrics).to_csv("metrics_softtopk_v3.csv", index=False)
plt.plot(train_loss_history, label="train")
plt.plot(val_loss_history, label="val")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train/Val Loss Curve")
plt.savefig("loss_curve_softtopk_v3.png", dpi=300)
plt.show()