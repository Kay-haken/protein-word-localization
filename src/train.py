# =============================
# 改造：加上 subsequence importance 加权 loss
# =============================
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random

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
# 1. Focal Loss (11 类)
# ----------------------------
def focal_loss(input, target, pos_weight=None, gamma=1.5):
    if pos_weight is not None:
        pos_weight = pos_weight.to(input.device)
    bce_loss = F.binary_cross_entropy_with_logits(
        input, target, pos_weight=pos_weight, reduction="none"
    )
    pt = torch.sigmoid(input) * target + (1 - torch.sigmoid(input)) * (1 - target)
    loss = ((1 - pt) ** gamma * bce_loss).mean(dim=1)  # per-sample loss
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

    def forward(self, x, mask, lengths, importance=None):
        B, L, D = x.shape
        x = x.view(B, L, self.n_heads, D // self.n_heads)
        x = self.preattn_ln(x)
        mul = (x * self.Q.weight.view(1, 1, self.n_heads, D // self.n_heads)).sum(-1)
        mul = mul.masked_fill(~mask.unsqueeze(-1), float("-1e9"))

        # ------------------------
        # ⭐ 加入 importance 影响 attention
        if importance is not None:
            # importance: (B, L)
            imp = importance.unsqueeze(-1)
            imp = imp / (imp.mean(dim=1, keepdim=True) + 1e-8)
            mul = mul + torch.log(imp + 1e-8)
        # ------------------------

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

    def forward(self, x, lengths, mask, importance=None):
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), mask], dim=1)
        lengths = lengths + 1

        # ------------------------
        # importance 同时加上 cls token
        if importance is not None:
            cls_imp = torch.ones(B, 1, device=x.device)
            importance = torch.cat([cls_imp, importance], dim=1)
        # ------------------------

        x = self.initial_ln(x)
        x = self.lin(x)
        x = F.relu(x)
        x = self.dropout(x)
        x_pool, x_attn = self.attn_head(x, mask, lengths, importance)
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

        self.embeddings, self.labels, self.masks, self.lengths, self.importance_list = [], [], [], [], []

        label_cols = [
            'Membrane', 'Cytoplasm', 'Nucleus', 'Extracellular', 'Cell membrane',
            'Mitochondrion', 'Plastid', 'Endoplasmic reticulum', 'Lysosome/Vacuole',
            'Golgi apparatus', 'Peroxisome'
        ]

        for k in self.keys:
            d = self.data_npz[k].item() if isinstance(self.data_npz[k], np.ndarray) else self.data_npz[k]
            emb = torch.tensor(d["embeddings"], dtype=torch.float32)
            imp = torch.tensor(d["importance"], dtype=torch.float32)  # ⭐ importance
            self.embeddings.append(emb)
            self.importance_list.append(imp)
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
        return self.embeddings[idx], self.importance_list[idx], self.lengths[idx], self.masks[idx], self.labels[idx], self.keys[idx], self.keys[idx]

# ----------------------------
# 5. Collate_fn
# ----------------------------
def collate_fn(batch):
    embeddings, importance_list, lengths, masks, labels, seq_names, ACCs = zip(*batch)
    embeddings_padded = pad_sequence(embeddings, batch_first=True)
    importance_padded = pad_sequence(importance_list, batch_first=True)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    masks_tensor = pad_sequence(masks, batch_first=True)
    labels_tensor = torch.stack(labels, dim=0)
    return embeddings_padded, importance_padded, lengths_tensor, masks_tensor, labels_tensor, seq_names, ACCs

# ----------------------------
# 6. 训练配置
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
label_csv = "/path/to/project/multisub_5_partitions_unique.csv"
pool_npz = "/path/to/project/sequence_embeddings_avg_pool_subseqs_with_importance_val.npz"

dataset = InMemoryNPZDataset(pool_npz, label_csv, max_samples=30000)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(seed)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

train_indices = train_dataset.indices
val_indices = val_dataset.indices
np.save("train_indices.npy", np.array(train_indices))
np.save("val_indices.npy", np.array(val_indices))
print(f"Train size: {len(train_indices)} saved -> train_indices.npy")
print(f"Val size: {len(val_indices)} saved -> val_indices.npy")

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=collate_fn)

model = BaseModel(embed_dim=1280, num_classes=11, dropout_rate=0.15).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3*1e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
)

n_epochs = 800
train_loss_history, val_loss_history, attention_dict, all_metrics = [], [], [], []
patience = 20
wait = 0
best_val_acc = 0

# ----------------------------
# 7. 训练循环（带 importance loss）
# ----------------------------
for epoch in range(n_epochs):
    print(f"\n===== Epoch {epoch + 1}/{n_epochs} =====")
    model.train()
    epoch_train_loss = 0
    for x, importance, lengths, mask, y, seq_names, ACCs in train_loader:
        x, importance, lengths, mask, y = x.to(device), importance.to(device), lengths.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        logits, attn = model(x, lengths, mask, importance)

        # ------------------------
        # ⭐ 使用 importance 对 loss 加权
        seq_imp = importance.mean(dim=1)  # (B,)
        seq_imp = seq_imp / (seq_imp.mean() + 1e-8)
        loss_vec = focal_loss(logits, y)
        loss = (loss_vec * seq_imp).mean()
        # ------------------------

        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_loss = epoch_train_loss / len(train_loader)
    train_loss_history.append(train_loss)

    # ----------------------------
    # 验证 (不使用 importance)
    model.eval()
    epoch_val_loss = 0
    y_true_all, y_pred_all = [], []
    attn_all = []
    with torch.no_grad():
        for x, importance, lengths, mask, y, seq_names, ACCs in val_loader:
            x, lengths, mask, y = x.to(device), lengths.to(device), mask.to(device), y.to(device)
            logits, attn = model(x, lengths, mask)
            loss = focal_loss(logits, y).mean()
            epoch_val_loss += loss.item()

            for i, name in enumerate(seq_names):
                attn_all.append({name: attn[i].cpu().numpy()})

            y_pred = torch.sigmoid(logits).cpu().numpy() > 0.5
            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(y_pred)

    val_loss = epoch_val_loss / len(val_loader)
    val_loss_history.append(val_loss)

    # --------------------
    # 计算指标
    y_true_all = np.vstack(y_true_all)
    y_pred_all = np.vstack(y_pred_all)

    acc_total = accuracy_score(y_true_all, y_pred_all)
    precision_total = precision_score(y_true_all, y_pred_all, average='macro', zero_division=0)
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

    # --------------------
    # 保存最佳模型 + Early Stopping
    if acc_total > best_val_acc:
        best_val_acc = acc_total
        wait = 0
        torch.save(model.state_dict(), "best_model_weights_with_importance.pth")
        torch.save(model, "best_full_model_with_importance.pth")
        print(f"Best model saved at epoch {epoch + 1} with val_accuracy {best_val_acc:.4f}")
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

# ----------------------------
# 8. 保存指标和 Attention
# ----------------------------
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("metrics_full_with_importance.csv", index=False)
print("训练指标已保存到 metrics_full_with_importance.csv")

attn_df = pd.DataFrame(attn_all)
attn_df.to_csv("attention_values_full_with_importance.csv", index=False)
print("注意力已保存到 attention_values_full_with_importance.csv")

plt.plot(range(1, len(train_loss_history)+1), train_loss_history, marker='o', label="Train Loss")
plt.plot(range(1, len(val_loss_history)+1), val_loss_history, marker='x', label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Curve")
plt.legend()
plt.savefig("loss_curve_full_with_importance.png", dpi=300)
plt.show()