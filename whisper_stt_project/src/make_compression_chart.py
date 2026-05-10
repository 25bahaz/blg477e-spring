"""Helper that draws a bar chart of size + PSNR for the report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(json_path: str, out_path: str) -> None:
    data = json.loads(Path(json_path).read_text())
    formats = [d["target_format"].upper() for d in data]
    sizes = [d["target_kb"] for d in data]
    ratios = [d["ratio"] for d in data]
    psnr = [d["psnr_db"] if np.isfinite(d["psnr_db"]) else 100.0 for d in data]
    src_kb = data[0]["source_kb"]

    x = np.arange(len(formats))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    bars1 = ax1.bar(x - width / 2, [src_kb] * len(formats), width,
                    label="Original WAV [kB]", color="#bbbbbb")
    bars2 = ax1.bar(x + width / 2, sizes, width,
                    label="Compressed [kB]", color="#1f77b4")
    ax1.set_ylabel("File size [kB]")
    ax1.set_xticks(x)
    ax1.set_xticklabels(formats)
    ax1.set_title("Compression: file size and round-trip PSNR")
    ax1.grid(axis="y", alpha=0.3)
    for b, r in zip(bars2, ratios):
        ax1.annotate(f"{r:.2f}×",
                     xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, psnr, "o-", color="#d62728", label="PSNR [dB]")
    ax2.set_ylabel("PSNR [dB]")
    ax2.set_ylim(0, max(psnr) * 1.1)

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/compression.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "figures/compression_chart.png"
    main(json_path, out_path)
