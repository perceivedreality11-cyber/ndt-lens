"""Render a publication-independent figure from recorded synthetic metrics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.summary.read_text(encoding="utf-8"))
    history = pd.read_csv(args.csv)
    history.columns = history.columns.str.strip()
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="#0c1220")
    for ax in axes:
        ax.set_facecolor("#101a2b")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.12)
    axes[0].plot(history["epoch"], history["metrics/mAP50(B)"], color="#20d6ad", label="val mAP50")
    axes[0].plot(history["epoch"], history["metrics/mAP50-95(B)"], color="#93b7ff", label="val mAP50-95")
    axes[0].set(
        xlabel="Epoch", ylabel="Validation metric", ylim=(0, 1.05), title="Learning on procedural phantoms"
    )
    axes[0].legend(loc="lower right", frameon=False)
    names = ["Precision", "Recall", "F1"]
    for offset, name, color in [(-0.18, "heuristic", "#93b7ff"), (0.18, "yolo", "#20d6ad")]:
        values = [metrics[name]["micro"][key] for key in ("precision", "recall", "f1")]
        positions = [i + offset for i in range(3)]
        axes[1].bar(
            positions,
            values,
            width=0.34,
            color=color,
            label="Morphology" if name == "heuristic" else "YOLO11n",
        )
        for x, value in zip(positions, values):
            axes[1].text(x, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    axes[1].set(xticks=range(3), xticklabels=names, ylim=(0, 1.14), title="Held-out test / fixed IoU = 0.5")
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle("NDT LENS  /  SYNTHETIC DATA ONLY", fontsize=17, fontweight="bold", color="#20d6ad", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "120 procedural images | 84 train / 18 val / 18 test | seed 2026 | CPU | not an industrial benchmark",
        ha="center",
        color="#a7b7cc",
        fontsize=9,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
