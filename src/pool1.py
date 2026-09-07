import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import argparse


# ==========================
# 参数解析
# ==========================
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seq_train",
        type=str,
        required=True
    )

    parser.add_argument(
        "--seq_val",
        type=str,
        required=True
    )

    # 原始 ESM embedding
    parser.add_argument(
        "--global_npz",
        type=str,
        required=True
    )


    parser.add_argument(
        "--npz_prefix",
        type=str,
        default="embeddings"
    )


    parser.add_argument(
        "--npz_count",
        type=int,
        default=10
    )


    parser.add_argument(
        "--target_layer",
        type=int,
        default=33
    )


    parser.add_argument(
        "--cycle",
        type=int,
        default=0,
        help="Recycle cycle index, start from 0"
    )


    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )


    return parser.parse_args()



# ==========================
# Load original ESM embeddings
# ==========================
def load_embeddings(
        npz_files,
        target_layer
):

    sequence_embeddings = {}


    for npz_file in npz_files:


        if not os.path.exists(npz_file):

            continue


        print(
            "Loading:",
            npz_file
        )


        data = np.load(
            npz_file,
            allow_pickle=True
        )


        seq_names = data["sequence_names"]


        per_tok_dict = (
            data["per_token_embeddings"]
            .item()
        )


        if target_layer not in per_tok_dict:

            continue


        layer_emb = (
            per_tok_dict[target_layer]
        )


        for name, emb in zip(
                seq_names,
                layer_emb
        ):

            sequence_embeddings[
                str(name)
            ] = emb



    print(
        "Loaded sequences:",
        len(sequence_embeddings)
    )


    return sequence_embeddings



# ==========================
# process split
# ==========================
def process_split(
        record_csv,
        sequence_embeddings,
        cycle
):


    df = pd.read_csv(record_csv)


    df["pos"] = df["pos"].apply(
        lambda x:
        [
            int(i)
            for i in str(x)
            .replace("[","")
            .replace("]","")
            .split(",")
            if i.strip()
        ]
    )


    df["sequence"] = (
        df["sequence"]
        .astype(str)
        .str.strip()
    )



    output_embeddings = {}

    order_records = []



    # ==========================
    # Dynamic recycle weights
    # ==========================

    # ==========================
    # Dynamic recycle weights
    # Cycle0: Raw=0.60  IG=0.40
    # Cycle1: Raw=0.70  IG=0.30
    # Cycle2: Raw=0.80  IG=0.20
    # Cycle3: Raw=0.90  IG=0.10
    # Cycle4+:Raw=1.00  IG=0.00
    # ==========================

    ig_weight = max(
        0.40 - 0.10 * cycle,
        0.0
    )

    raw_weight = 1.0 - ig_weight

    print(
        f"Cycle {cycle}: "
        f"Raw weight={raw_weight:.2f}, "
        f"IG weight={ig_weight:.2f}"
    )



    grouped = df.groupby(
        "sequence"
    )



    for seq_name, group in tqdm(
        grouped,
        desc=f"Processing {record_csv}"
    ):



        if seq_name not in sequence_embeddings:

            continue



        # ==================================================
        # Fixed original ESM embedding
        # ==================================================

        esm_layer = (
            sequence_embeddings[seq_name]
        )



        # ==================================================
        # IG normalization
        # ==================================================

        ig_raw = (
            group["importance"]
            .astype(np.float32)
            .values
        )


        # remove negative

        ig_raw = np.maximum(
            ig_raw,
            0
        )


        # log stabilization

        ig_raw = np.log1p(
            ig_raw
        )



        ig_min = ig_raw.min()

        ig_max = ig_raw.max()



        if ig_max > ig_min:

            ig_norm = (
                ig_raw - ig_min
            ) / (
                ig_max - ig_min + 1e-8
            )


        else:

            ig_norm = np.zeros_like(
                ig_raw
            )



        # keep IG scale

        ig_norm = ig_norm * 4.0




        subseq_embeddings = []

        importance_list = []

        local_records = []



        # ==================================================
        # subsequence processing
        # ==================================================

        for idx,row in enumerate(
                group.itertuples(index=False)
        ):


            pos_array = np.array(
                row.pos
            )


            pos_array = pos_array[
                pos_array < esm_layer.shape[0]
            ]



            if len(pos_array)==0:

                continue



            token_emb = torch.tensor(
                esm_layer[pos_array,:],
                dtype=torch.float32
            )


            avg_emb = (
                token_emb.mean(dim=0)
            )



            # ==================================================
            # Fixed Raw importance
            # ==================================================

            raw_importance = (
                avg_emb.norm()
                .item()
            )


            raw_importance = np.log1p(
                max(raw_importance,0)
            )



            # ==================================================
            # Current cycle IG
            # ==================================================

            ig_importance = float(
                ig_norm[idx]
            )



            # ==================================================
            # Dynamic Fusion
            # ==================================================

            importance = (
                raw_weight * raw_importance
                +
                ig_weight * ig_importance
            )



            subseq_embeddings.append(
                avg_emb
            )


            importance_list.append(
                importance
            )



            local_records.append(

                {

                "sequence":
                seq_name,


                "rank":
                len(subseq_embeddings)-1,


                "pos_start":
                int(pos_array[0]),


                "raw_importance":
                raw_importance,


                "ig_importance":
                ig_importance,


                "raw_weight":
                raw_weight,


                "ig_weight":
                ig_weight,


                "importance":
                importance,


                "pos_list":
                pos_array.tolist()

                }

            )



        if len(subseq_embeddings)==0:

            continue



        # keep position order

        sorted_idx = np.argsort(
            [
                x["pos_start"]
                for x in local_records
            ]
        )



        stacked_embeddings = torch.stack(
            [
                subseq_embeddings[i]
                for i in sorted_idx
            ]
        )



        stacked_importance = np.array(
            [
                importance_list[i]
                for i in sorted_idx
            ]
        )



        output_embeddings[seq_name] = {

            "embeddings":
            stacked_embeddings.cpu().numpy(),


            "importance":
            stacked_importance
        }



        order_records.extend(
            local_records
        )



    return (
        output_embeddings,
        order_records
    )



# ==========================
# main
# ==========================
def main():

    args = parse_args()


    os.makedirs(
        args.output_dir,
        exist_ok=True
    )



    npz_files = [

        os.path.join(
            args.global_npz,
            f"{args.npz_prefix}{i}_per_tok.npz"
        )

        for i in range(
            1,
            args.npz_count+1
        )

    ]



    # always original ESM

    sequence_embeddings = load_embeddings(
        npz_files,
        args.target_layer
    )



    splits = {

        "train":
        args.seq_train,


        "val":
        args.seq_val

    }



    for split_name,record_csv in splits.items():


        if not os.path.exists(record_csv):

            print(
                "Missing:",
                record_csv
            )

            continue



        output_embeddings, order_records = process_split(
            record_csv,
            sequence_embeddings,
            args.cycle
        )



        output_npz = os.path.join(

            args.output_dir,

            f"sequence_embeddings_avg_pool_subseqs_with_importance_{split_name}.npz"

        )



        output_csv = os.path.join(

            args.output_dir,

            f"sequence_stack_order_subseqs_{split_name}.csv"

        )



        np.savez_compressed(
            output_npz,
            **output_embeddings
        )


        pd.DataFrame(
            order_records
        ).to_csv(
            output_csv,
            index=False
        )



        print(
            f"[{split_name}] saved:",
            output_npz
        )


        print(
            f"[{split_name}] saved:",
            output_csv
        )




if __name__=="__main__":

    main()
