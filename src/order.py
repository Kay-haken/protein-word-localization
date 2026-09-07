from recycle import RecyclePipeline

config = {
    "filtered_csv": "/path/to/project/filtered_positions_de.csv",
    "npz_embeddings": "/path/to/project/filtered_positions_de.npz",
    "label_csv": "/path/to/project/multisub_5_partitions_unique.csv",
    "global_npz": "/path/to/project",

    "output_dir": "./recycle_results2",

    "max_recycles": 3,

    "index_dir": "./split_index2",

    "lr": 3e-5,
    "batch_size": 64,
    "max_epochs": 800,
    "dropout": 0.15,
    "seed": 65,
    "patience": 20,
    "weighted_decay": 1e-4
}

pipeline = RecyclePipeline(config)
pipeline.run_all_cycles()
