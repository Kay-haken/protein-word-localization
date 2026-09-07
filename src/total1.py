import os
import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================
# 路径
# =========================

train_dir = (
    "/path/to/project/recycle_results7/cycle_1/ig_full/train/correct_class_residue"
)

val_dir = (
    "/path/to/project/recycle_results/cycle_1/ig_full/val/correct_class_residue"
)


filtered_csv = (
    "/path/to/project/filtered_positions.csv"
)


out_dir = (
    "/path/to/project/IG_class_analysis"
)


os.makedirs(
    out_dir,
    exist_ok=True
)



# =========================
# 1. 建立 embedding index 映射
# =========================

df = pd.read_csv(filtered_csv)



df["pos"] = df["pos"].apply(

    lambda x:
    [
        int(i)

        for i in str(x)
        .replace("[","")
        .replace("]","")
        .split()

        if i.strip()
    ]

)


df["chain"] = (
    df["chain"]
    .astype(str)
    .str.strip()
)



mapping_rows=[]



for chain,group in tqdm(

        df.groupby("chain"),

        desc="Building mapping"

):


    tmp = sorted(

        group.to_dict("records"),

        key=lambda x:x["pos"][0]

    )



    for idx,item in enumerate(tmp):


        mapping_rows.append({

            "sequence":chain,

            "embedding_index":idx,

            "pos":item["pos"],

            "seq_fragment":item["seq"]

        })



mapping_df = pd.DataFrame(
    mapping_rows
)


print(
    "Mapping:",
    len(mapping_df)
)



# =========================
# 2. 分析 train / val
# =========================


def process_split(root_dir, split):


    rows=[]



    for cls in os.listdir(root_dir):


        cls_dir=os.path.join(

            root_dir,

            cls

        )


        if not os.path.isdir(cls_dir):
            continue



        print(
            "Processing",
            split,
            cls
        )



        for file in tqdm(

            os.listdir(cls_dir)

        ):


            if not file.endswith(".npz"):
                continue



            seq=file.replace(
                ".npz",
                ""
            )



            path=os.path.join(

                cls_dir,

                file

            )


            data=np.load(path)



            fusion=data["fusion"]



            # importance 排序

            sorted_idx=np.argsort(

                fusion

            )[::-1]



            for rank,pos in enumerate(

                    sorted_idx,

                    start=1

            ):


                rows.append({

                    "split":split,

                    "class":cls,

                    "sequence":seq,

                    "rank":rank,

                    "embedding_index":
                        int(pos),

                    "importance":
                        float(
                            fusion[pos]
                        )

                })



    result=pd.DataFrame(rows)



    # 加 residue word

    result=result.merge(

        mapping_df,

        on=[

            "sequence",

            "embedding_index"

        ],

        how="left"

    )



    # 保存完整位置

    result.to_csv(

        os.path.join(

            out_dir,

            f"{split}_all_IG_position_sequence.csv"

        ),

        index=False

    )



    print(

        split,

        "positions:",

        len(result)

    )



    # =====================
    # 频率统计 + importance
    # =====================


    freq=(

        result

        .groupby(

            [

                "class",

                "seq_fragment"

            ]

        )

        .agg(


            count=(

                "seq_fragment",

                "size"

            ),


            mean_importance=(

                "importance",

                "mean"

            ),


            max_importance=(

                "importance",

                "max"

            ),


            importance_sum=(

                "importance",

                "sum"

            )


        )

        .reset_index()

    )



    freq=freq.sort_values(

        [

            "class",

            "count"

        ],

        ascending=[

            True,

            False

        ]

    )



    freq.to_csv(

        os.path.join(

            out_dir,

            f"{split}_fragment_frequency.csv"

        ),

        index=False

    )



    return freq





# =========================
# train val
# =========================


train_freq=process_split(

    train_dir,

    "train"

)



val_freq=process_split(

    val_dir,

    "val"

)



# =========================
# train + val 合并
# =========================


merge=(


    train_freq.rename(

        columns={


            "count":
            "train_count",


            "mean_importance":
            "train_mean_importance",


            "max_importance":
            "train_max_importance",


            "importance_sum":
            "train_importance_sum"

        }

    )


    .merge(


        val_freq.rename(

            columns={


                "count":
                "val_count",


                "mean_importance":
                "val_mean_importance",


                "max_importance":
                "val_max_importance",


                "importance_sum":
                "val_importance_sum"


            }

        ),


        on=[

            "class",

            "seq_fragment"

        ],


        how="outer"


    )


)



merge=merge.fillna(0)



merge["train_count"]=(
    merge["train_count"]
    .astype(int)
)


merge["val_count"]=(
    merge["val_count"]
    .astype(int)
)



merge["total_count"]=(

    merge["train_count"]

    +

    merge["val_count"]

)



merge["total_importance_sum"]=(

    merge["train_importance_sum"]

    +

    merge["val_importance_sum"]

)



merge=merge.sort_values(

    [

        "class",

        "total_count"

    ],

    ascending=[

        True,

        False

    ]

)



merge.to_csv(

    os.path.join(

        out_dir,

        "train_val_fragment_frequency_total.csv"

    ),

    index=False

)



print("================")
print("DONE")
print("================")

print(
    merge.head(20))