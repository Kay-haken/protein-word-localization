import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import os

# =========================
# 文件路径
# =========================
base_dir = "/path/to/project/"
npz_files = [os.path.join(base_dir, f"embeddings{i}_per_tok.npz") for i in range(1, 11)]

record_csv = os.path.join(base_dir, "SEQ_total_deduplicated.csv")

output_npz = os.path.join(base_dir, "sequence_embeddings_avg_pool_subseqs_with_importance.npz")
output_order_csv = os.path.join(base_dir, "sequence_stack_order_subseqs.csv")

target_layer = 33

# =========================
# 读取 record 文件
# =========================
df_csv = pd.read_csv(record_csv)

# pos -> list[int]
df_csv["pos"] = df_csv["pos"].apply(
    lambda x: [int(i) for i in str(x).replace("[", "").replace("]", "").split(",") if i.strip()]
)

df_csv["sequence"] = df_csv["sequence"].astype(str).str.strip()

# =========================
# 读取 embedding npz
# =========================
sequence_embeddings = {}

for npz_file in npz_files:
    if not os.path.exists(npz_file):
        continue

    data = np.load(npz_file, allow_pickle=True)
    sequence_names = data["sequence_names"]
    per_token_embeddings_dict = data["per_token_embeddings"].item()

    if target_layer not in per_token_embeddings_dict:
        continue

    layer_embeddings = per_token_embeddings_dict[target_layer]

    for name, emb in zip(sequence_names, layer_embeddings):
        sequence_embeddings[name] = emb

# =========================
# 输出容器
# =========================
output_embeddings = {}
order_records = []

# =========================
# 按 sequence 处理
# =========================
grouped = df_csv.groupby("sequence")

for seq_name, group in tqdm(grouped, desc="Processing sequences"):

    if seq_name not in sequence_embeddings:
        continue

    emb_layer = sequence_embeddings[seq_name]  # (L, D)
    subseq_embeddings = []
    importance_list = []

    for row in group.itertuples(index=False):
        pos_array = np.array(row.pos)
        pos_array = pos_array[pos_array < emb_layer.shape[0]]  # 越界检查
        if len(pos_array) == 0:
            continue

        token_emb = torch.tensor(emb_layer[pos_array, :], dtype=torch.float32)
        avg_emb = token_emb.mean(dim=0)  # 平均向量
        importance = float(avg_emb.norm().item())

        subseq_embeddings.append(avg_emb)
        importance_list.append(importance)

        # 保存顺序信息
        order_records.append({
            "sequence": seq_name,
            "rank": len(subseq_embeddings) - 1,  # 当前顺序
            "pos_start": int(pos_array[0]),
            "importance": importance,
            "pos_list": pos_array.tolist()
        })

    if len(subseq_embeddings) == 0:
        continue

    # 按 pos_start 排序
    sorted_idx = np.argsort([item["pos_start"] for item in order_records if item["sequence"] == seq_name])
    stacked_embeddings = torch.stack([subseq_embeddings[i] for i in sorted_idx])
    stacked_importance = np.array([importance_list[i] for i in sorted_idx])

    # 保存到 NPZ
    output_embeddings[seq_name] = {
        "embeddings": stacked_embeddings.cpu().numpy(),
        "importance": stacked_importance
    }

# =========================
# 保存 NPZ
# =========================
np.savez_compressed(output_npz, **output_embeddings)
print(f"Saved embeddings + importance NPZ -> {output_npz}")

# =========================
# 保存排序/重要性 CSV
# =========================
pd.DataFrame(order_records).to_csv(output_order_csv, index=False)
print(f"Saved order CSV -> {output_order_csv}")