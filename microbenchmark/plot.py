#!/usr/bin/env python3
"""Pure-IO random read bandwidth vs. number of SMs (grouped bar chart)."""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_csv(path):
    sms, bws = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sms.append(int(row["sms"]))
            bws.append(float(row["bandwidth_gibs"]))
    return sms, bws

def main():
    systems = [
        ("BaM",       os.path.join(SCRIPT_DIR, "results_bam_read_bw.csv")),
        ("CoPilotIO", os.path.join(SCRIPT_DIR, "results_copilotio_read_bw.csv")),
    ]

    colors = {"BaM": "#1f77b4", "CoPilotIO": "#2ca02c"}

    data = {}
    sm_ticks = None
    for label, path in systems:
        if not os.path.exists(path):
            print(f"Skip {label}: {path} not found")
            continue
        sms, bws = load_csv(path)
        data[label] = bws
        if sm_ticks is None:
            sm_ticks = sms

    if not data or sm_ticks is None:
        print("No data to plot")
        return

    n_groups = len(sm_ticks)
    n_systems = len(data)
    bar_width = 0.6 / n_systems
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    for i, (label, bws) in enumerate(data.items()):
        offset = (i - (n_systems - 1) / 2) * bar_width
        bars = ax.bar(x + offset, bws, bar_width * 0.9,
                      label=label, color=colors.get(label, None), edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, bws):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Number of SMs", fontsize=12)
    ax.set_ylabel("Bandwidth (GiB/s)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sm_ticks])
    ax.set_ylim(0, max(max(v) for v in data.values()) * 1.3)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()

    for fmt in ["pdf", "png"]:
        out = os.path.join(SCRIPT_DIR, f"read_bw.{fmt}")
        fig.savefig(out, dpi=300 if fmt == "pdf" else 150)
        print(f"Saved to {out}")

if __name__ == "__main__":
    main()
