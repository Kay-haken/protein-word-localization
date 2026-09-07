import os
import json
from pathlib import Path
import shutil
import subprocess
import sys

PYTHON = sys.executable


class RecyclePipeline:
    def __init__(self, config):
        """
        config: dict 必须包含
            - filtered_csv
            - npz_embeddings
            - label_csv
            - global_npz
            - output_dir
            - max_recycles (int)
            - recycle_steps (list[int]): 哪些 step 可以回流, e.g., [1,2,3]
        """
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_recycles = config.get("max_recycles", 3)
        self.recycle_steps = config.get("recycle_steps", [1, 2, 3])
        self.state_file = self.output_dir / "pipeline_state.json"
        self.state = self.load_state()

    def run_cmd(self, name, cmd):
        print(f"\n>>> [{name}] Running command:\n{cmd}")

        result = subprocess.run(
            cmd,
            shell=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Step failed: {name}")

    def load_state(self):
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {"cycle": 0, "history": []}

    def save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    # =========================
    # Run command
    # =========================

    def cycle_dir(self, cycle):
        return self.output_dir / f"cycle_{cycle}"

    # =========================
    # Run a single cycle
    # =========================
    def run_cycle(self, cycle):
        cdir = self.cycle_dir(cycle)
        cdir.mkdir(parents=True, exist_ok=True)
        final_model = (
                cdir /
                "train_output" /
                "recycle_model_weights.pth"
        )

        if final_model.exists():
            print(
                f"Cycle {cycle} already completed, skip"
            )
            return

        print(f"\n==============================")
        print(f"🔥 Cycle {cycle + 1}/{self.max_recycles} start")
        print(f"==============================")

        index_dir = self.config.get('index_dir', None)
        index_option = f"--index_dir {index_dir}" if index_dir else ""

        # =========================
        # STEP 1: step.py (第一轮)
        # =========================

        if cycle == 0:
            model_path = cdir / "best_model_weights.pth"

            cmd_step = (
                f"{PYTHON} step.py "
                f"--npz_embeddings {self.config['npz_embeddings']} "
                f"--label_csv {self.config['label_csv']} "
                f"--output_dir {cdir} "
                f"{index_option} "
                f"--lr {self.config.get('lr', 3e-5)} "
                f"--batch_size {self.config.get('batch_size', 64)} "
                f"--max_epochs {self.config.get('max_epochs', 800)} "
                f"--dropout {self.config.get('dropout', 0.15)} "
                f"--seed {self.config.get('seed', 42)} "
                f"--patience {self.config.get('patience', 20)} "
                f"--weighted_decay {self.config.get('weighted_decay', 1e-4)}"
            )

            if model_path.exists():
                print("STEP1 already done, skip")
            else:
                self.run_cmd("step_train", cmd_step)

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Step model not found: {model_path}"
                )

        else:
            prev_train_dir = self.cycle_dir(cycle - 1) / "train_output"

            model_path = (
                    prev_train_dir /
                    "recycle_model_weights.pth"
            )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"上轮模型不存在: {model_path}"
                )

        # =========================
        # STEP 2: IG + ANALYZE

        ig_dir = cdir / "IG_30K_RESULTS"

        # IG 使用不同 npz：第一轮是原始 embedding，之后是上一轮 pool1 输出
        ig_npz = self.config['npz_embeddings']

        if not Path(ig_npz).exists():
            raise FileNotFoundError(
                f"Embedding not found: {ig_npz}"
            )
        print(f"\n[Cycle {cycle}]")
        print(f"Model: {model_path}")
        print(f"Embedding: {ig_npz}")
        # =========================
        # STEP 2: IG + ANALYZE
        # =========================

        ig_dir = cdir / "IG_30K_RESULTS"

        if cycle == 0:

            ig_script = "ig_step.py"

            cmd_ig = (
                f"{PYTHON} {ig_script} "
                f"--model_pth {model_path} "
                f"--npz_embeddings {self.config['npz_embeddings']} "
                f"--label_csv {self.config['label_csv']} "
                f"--output_dir {ig_dir} "
                f"{index_option}"
            )

        else:

            ig_script = "ig_recycle.py"

            prev_dir = self.cycle_dir(cycle - 1)

            train_npz = (
                    prev_dir /
                    "sequence_embeddings_avg_pool_subseqs_with_importance_train.npz"
            )

            val_npz = (
                    prev_dir /
                    "sequence_embeddings_avg_pool_subseqs_with_importance_val.npz"
            )

            cmd_ig = (
                f"{PYTHON} {ig_script} "
                f"--model_pth {model_path} "
                f"--train_npz {train_npz} "
                f"--val_npz {val_npz} "
                f"--label_csv {self.config['label_csv']} "
                f"--output_dir {ig_dir} "
                f"--mode both "
            )
        ig_check = ig_dir / "train"

        if ig_check.exists() and len(list(ig_check.glob("fusion_npz/*.npz"))) > 0:
            print("IG already done, skip")
        else:
            self.run_cmd("IG", cmd_ig)
        for split in ["train", "val"]:

            all_csv = cdir / f"PerSequence_AllRankedImportance_{split}.csv"
            top10_csv = cdir / f"PerSequence_Top10Importance_{split}.csv"

            if all_csv.exists() and top10_csv.exists():
                print(f"Analyze {split} already done, skip")
                continue

            cmd_analyze = (
                f"{PYTHON} analyze1.py "
                f"--ig_dir {ig_dir} "
                f"--split {split} "
                f"--all_csv {all_csv} "
                f"--top10_csv {top10_csv}"
            )

            self.run_cmd(f"Analyze_{split}", cmd_analyze)

        # =========================
        # STEP 3: POS MAP
        # =========================
        cmd_pos = (
            f"{PYTHON} pos1.py "
            f"--filtered_csv {self.config['filtered_csv']} "
            f"--importance_dir {cdir} "
            f"--output_dir {cdir}"
        )
        seq_train_file = cdir / "SEQ_total_train.csv"
        seq_val_file = cdir / "SEQ_total_val.csv"

        if seq_train_file.exists() and seq_val_file.exists():
            print("POS already done, skip")
        else:
            self.run_cmd("pos1", cmd_pos)

        # =========================
        # STEP 4: POOLING
        # =========================
        cmd_pool = (
            f"{PYTHON} pool1.py "
            f"--seq_train {cdir}/SEQ_total_train.csv "
            f"--seq_val {cdir}/SEQ_total_val.csv "
            f"--global_npz {self.config['global_npz']} "
            f"--output_dir {cdir} "
            f"--cycle {cycle} "
        )
        pool_train_npz = (
                cdir /
                "sequence_embeddings_avg_pool_subseqs_with_importance_train.npz"
        )

        pool_val_npz = (
                cdir /
                "sequence_embeddings_avg_pool_subseqs_with_importance_val.npz"
        )

        if pool_train_npz.exists() and pool_val_npz.exists():
            print("POOL already done, skip")
        else:
            self.run_cmd("pool1", cmd_pool)

        train_npz = os.path.join(cdir, "sequence_embeddings_avg_pool_subseqs_with_importance_train.npz")
        val_npz = os.path.join(cdir, "sequence_embeddings_avg_pool_subseqs_with_importance_val.npz")
        label_csv = self.config["label_csv"]  # 使用原始 label CSV

        train_output_dir = os.path.join(cdir, "train_output")
        os.makedirs(train_output_dir, exist_ok=True)
        # =========================
        # STEP 5: train.py
        # =========================
        cmd_train_py = (
            f"{PYTHON} train1new.py "
            f"--train_npz {train_npz} "
            f"--val_npz {val_npz} "
            f"--label_csv {label_csv} "
            f"--output_dir {train_output_dir} "
            f"--model_out recycle_model_weights.pth "
            f"--embed_dim 1280 "
            f"--num_classes 11 "
            f"--batch_size 64 "
            f"--lr 3e-5 "
            f"--weight_decay 1e-4 "
            f"--epochs 200 "
            f"--patience 20 "
            f"--topk 5 "
            f"--tau 0.7 "
            f"--seed 60 "
            f"--num_workers 4 "
        )
        recycle_model = (
                Path(train_output_dir)
                / "recycle_model_weights.pth"
        )

        if recycle_model.exists():
            print("TRAIN already done, skip")
        else:
            self.run_cmd("train1new_py", cmd_train_py)

        if not recycle_model.exists():
            raise FileNotFoundError(
                f"Recycle model not found: {recycle_model}"
            )

        # =========================
        # STEP 6: 生成 recycle 权重
        # =========================

        # =========================
        # Record cycle
        # =========================
        self.state["history"].append({
            "cycle": cycle,
            "dir": str(cdir),
            "ig": str(ig_dir),

        })
        self.save_state()

        print(f"🔥 Cycle {cycle + 1} done")

    # =========================
    # Run all cycles
    # =========================
    def run_all_cycles(self):
        for cycle in range(self.max_recycles):
            self.run_cycle(cycle)