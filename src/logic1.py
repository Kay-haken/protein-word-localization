import os
import re
import torch
import numpy as np
import pandas as pd

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
matthews_corrcoef
)

from torch.utils.data import DataLoader

from model1 import (
InMemoryNPZDataset,
collate_fn,
BaseModel
)

# =====================================

# 路径

# =====================================

MODEL_PTH = "/path/to/project/recycle_results/cycle_0/best_model_weights.pth"

TEST_NPZ = "/path/to/project/sequence_embeddings_avg_hpafull.npz"

LABEL_CSV = "/path/to/project/hpa_testset.csv"

OUTPUT_DIR = "recycle_results/cycle_0"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================

# device

# =====================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================

# Dataset

# =====================================

dataset = InMemoryNPZDataset(
TEST_NPZ,
LABEL_CSV
)

loader = DataLoader(
dataset,
batch_size=64,
shuffle=False,
num_workers=4,
pin_memory=True,
collate_fn=collate_fn
)

# =====================================

# Model

# =====================================

model = BaseModel(
embed_dim=1280,
num_classes=11,
dropout_rate=0.15
).to(device)

state = torch.load(
MODEL_PTH,
map_location=device
)

model.load_state_dict(state)

model.eval()

# =====================================

# inference

# =====================================

y_true_all = []
y_pred_all = []




with torch.no_grad():

    for x, lengths, mask, y, seq_names, ACCs in loader:

        x = x.to(device)
        lengths = lengths.to(device)
        mask = mask.to(device)

        logits, attn = model(
            x,
            lengths,
            mask
        )

        pred = (
            torch.sigmoid(logits)
            > 0.5
        ).cpu().numpy()

        y_pred_all.append(pred)
        y_true_all.append(y.numpy())


# =====================================

# metrics

# =====================================

y_true_all = np.vstack(y_true_all)
y_pred_all = np.vstack(y_pred_all)

acc_total = accuracy_score(
y_true_all,
y_pred_all
)

precision_total = precision_score(
y_true_all,
y_pred_all,
average="macro",
zero_division=0
)

recall_total = recall_score(
y_true_all,
y_pred_all,
average="macro",
zero_division=0
)

macro_f1 = f1_score(
y_true_all,
y_pred_all,
average="macro",
zero_division=0
)

micro_f1 = f1_score(
y_true_all.flatten(),
y_pred_all.flatten(),
zero_division=0
)

total_mcc = matthews_corrcoef(
y_true_all.flatten(),
y_pred_all.flatten()
)

metrics = {
"accuracy": acc_total,
"precision": precision_total,
"recall": recall_total,
"macro_f1": macro_f1,
"micro_f1": micro_f1,
"total_mcc": total_mcc
}

# =====================================

# per-class metrics

# =====================================
for i in range(y_true_all.shape[1]):

    metrics[f"class_{i}_acc"] = accuracy_score(
        y_true_all[:, i],
        y_pred_all[:, i]
    )

    metrics[f"class_{i}_f1"] = f1_score(
        y_true_all[:, i],
        y_pred_all[:, i],
        zero_division=0
    )

    metrics[f"class_{i}_mcc"] = matthews_corrcoef(
        y_true_all[:, i],
        y_pred_all[:, i]
    )

# =====================================

# save

# =====================================

out_csv = os.path.join(
OUTPUT_DIR,
"hpa_test_metrics.csv"
)

pd.DataFrame([metrics]).to_csv(
out_csv,
index=False
)

print("====================================")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("====================================")
print("Saved ->", out_csv)