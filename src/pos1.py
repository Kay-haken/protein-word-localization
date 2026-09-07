import pandas as pd
from tqdm import tqdm
import argparse
import os

# --------------------------
# 参数解析
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Map sequence positions to embeddings with importance scores")

    parser.add_argument(
        "--filtered_csv",
        type=str,
        required=True,
        help="filtered_positions.csv 的路径"
    )

    parser.add_argument(
        "--importance_dir",
        type=str,
        required=True,
        help="存放 PerSequence_AllRankedImportance_{split}.csv 的目录"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录，最终 SEQ_total_{split}.csv 会生成在此目录"
    )

    return parser.parse_args()


# --------------------------
# 主函数
# --------------------------
def run_pos_mapping(filtered_csv, importance_csv, output_csv):
    # 1. 读取 filtered_positions.csv
    df = pd.read_csv(filtered_csv)

    df["pos"] = df["pos"].apply(
        lambda x: [int(i) for i in str(x).replace("[", "").replace("]", "").split()]
    )
    df["chain"] = df["chain"].str.strip()

    # 2. 构建 embedding_index -> seq 映射
    rows = []
    for chain_name, group in tqdm(df.groupby("chain"), desc="Building mapping"):
        tmp = sorted(group.to_dict("records"), key=lambda x: x["pos"][0])
        for idx, item in enumerate(tmp):
            rows.append({
                "sequence": chain_name,
                "embedding_index": idx,
                "pos": item["pos"],
                "seq_fragment": item["seq"]
            })

    mapping_df = pd.DataFrame(rows)
    mapping_df = mapping_df.drop_duplicates(subset=["sequence", "embedding_index"])

    # 3. 读取 importance CSV
    importance_df = pd.read_csv(importance_csv)
    importance_df["sequence"] = importance_df["sequence"].str.strip()

    # 4. merge
    merged = importance_df.merge(
        mapping_df,
        left_on=["sequence", "position"],
        right_on=["sequence", "embedding_index"],
        how="left"
    )
    merged = merged.drop_duplicates(subset=["sequence", "position"])

    # 5. 保存
    merged.to_csv(output_csv, index=False)
    print(f"Merged saved -> {output_csv}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    splits = ["train", "val"]
    for split in splits:
        importance_csv = os.path.join(
            args.importance_dir,
            f"PerSequence_AllRankedImportance_{split}.csv"
        )
        output_csv = os.path.join(
            args.output_dir,
            f"SEQ_total_{split}.csv"
        )

        print(f"\n==== Processing {split} ====")
        run_pos_mapping(args.filtered_csv, importance_csv, output_csv)


if __name__ == "__main__":
    main()