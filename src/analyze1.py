import os
import numpy as np
import pandas as pd
import argparse

# --------------------------
# 参数解析
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Compute ranked importance from fusion npz files")

    parser.add_argument(
        "--ig_dir",
        type=str,
        required=True,
        help="IG_30K_RESULTS 目录"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="val",
        help="选择处理数据集: train 或 val"
    )
    parser.add_argument(
        "--all_csv",
        type=str,
        default=None,
        help="保存全部位置排名的 CSV 文件名, 默认自动生成"
    )
    parser.add_argument(
        "--top10_csv",
        type=str,
        default=None,
        help="保存 Top10 位置的 CSV 文件名, 默认自动生成"
    )

    return parser.parse_args()

# --------------------------
# 主函数
# --------------------------
def main():
    args = parse_args()

    fusion_dir = os.path.join(args.ig_dir, args.split, "fusion_npz")

    if not os.path.exists(fusion_dir):
        raise ValueError(f"Fusion 目录不存在: {fusion_dir}")

    all_csv = args.all_csv or f"PerSequence_AllRankedImportance_{args.split}.csv"
    top10_csv = args.top10_csv or f"PerSequence_Top10Importance_{args.split}.csv"

    all_rows = []
    top10_rows = []

    for file in os.listdir(fusion_dir):
        if not file.endswith(".npz"):
            continue

        seq_name = file.replace(".npz", "")
        data = np.load(os.path.join(fusion_dir, file))
        fusion = data["fusion"]

        # 从大到小排序
        sorted_idx = np.argsort(fusion)[::-1]

        # ========= 全部位置 =========
        for rank, pos in enumerate(sorted_idx, start=1):
            all_rows.append({
                "sequence": seq_name,
                "rank": rank,
                "position": int(pos),
                "importance": float(fusion[pos])
            })

        # ========= Top10 =========
        for rank, pos in enumerate(sorted_idx[:10], start=1):
            top10_rows.append({
                "sequence": seq_name,
                "rank": rank,
                "position": int(pos),
                "importance": float(fusion[pos])
            })

    # --------------------------
    # 保存全部位置
    # --------------------------
    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(all_csv, index=False)

    # --------------------------
    # 保存 Top10
    # --------------------------
    top10_df = pd.DataFrame(top10_rows)
    top10_df.to_csv(top10_csv, index=False)

    print("全部位置:", len(all_df))
    print("Top10:", len(top10_df))
    print("\nTop10 示例:")
    print(top10_df.head(20))

# --------------------------
# 入口
# --------------------------
if __name__ == "__main__":
    main()