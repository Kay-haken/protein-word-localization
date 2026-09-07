import matplotlib.pyplot as plt


def draw_paper_pipeline():

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")

    # =========================
    # Layer 1: Input
    # =========================
    ax.text(
        0.1, 0.85,
        "Protein Sequences\n(NPZ Embeddings)",
        ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="#A6CEE3")
    )

    # =========================
    # Layer 2: Base Model
    # =========================
    ax.text(
        0.3, 0.85,
        "step.py\nBase Model Training",
        ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="#1F78B4", alpha=0.8, color="white")
    )

    # =========================
    # Layer 3: Explainability
    # =========================
    ax.text(
        0.55, 0.9,
        "IG",
        ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="#B2DF8A")
    )

    ax.text(
        0.55, 0.82,
        "Attention",
        ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="#33A02C")
    )

    ax.text(
        0.55, 0.74,
        "GradNorm",
        ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="#FB9A99")
    )

    # fusion
    ax.text(
        0.75, 0.82,
        "Fusion Importance",
        ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="#E31A1C", alpha=0.8)
    )

    # =========================
    # Downstream
    # =========================
    ax.text(
        0.75, 0.6,
        "analyze1.py\nranking",
        ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="#FDBF6F")
    )

    ax.text(
        0.55, 0.6,
        "pos1.py\nmapping",
        ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="#FF7F00")
    )

    ax.text(
        0.35, 0.6,
        "pool1.py\nimportance pooling",
        ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="#CAB2D6")
    )

    ax.text(
        0.15, 0.6,
        "train1new.py\nre-training",
        ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="#6A3D9A", color="white")
    )

    # =========================
    # Recycled model
    # =========================
    ax.text(
        0.15, 0.4,
        "Recycled Model",
        ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="#FFFF99")
    )

    # =========================
    # arrows (main flow)
    # =========================
    def arrow(x1, y1, x2, y2, color="black"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=2, color=color)
        )

    arrow(0.18, 0.85, 0.25, 0.85)
    arrow(0.35, 0.85, 0.48, 0.85)

    # IG flow
    arrow(0.35, 0.85, 0.52, 0.88, "green")
    arrow(0.35, 0.85, 0.52, 0.82, "green")
    arrow(0.35, 0.85, 0.52, 0.76, "green")

    arrow(0.6, 0.82, 0.72, 0.82, "red")

    arrow(0.75, 0.78, 0.75, 0.63)
    arrow(0.65, 0.6, 0.65, 0.6)
    arrow(0.45, 0.6, 0.45, 0.6)
    arrow(0.25, 0.6, 0.25, 0.6)

    # recycle loop
    ax.annotate(
        "",
        xy=(0.15, 0.55),
        xytext=(0.15, 0.45),
        arrowprops=dict(arrowstyle="->", lw=2, color="blue")
    )

    ax.text(0.17, 0.5, "cycle", color="blue")

    plt.title("IG × Attention × GradNorm Recycle Framework", fontsize=16)
    plt.tight_layout()
    plt.savefig("figure1_pipeline_paper.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    draw_paper_pipeline()