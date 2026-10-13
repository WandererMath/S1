#!/usr/bin/env python3
"""Plot counts of experimental S1 classes for every consensus cluster."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "5-6-7-8"
ORDER = ["S1_dependent", "S1_unaffected", "S1_repressed_or_acetyl_preferring"]
COLOURS = ["#D55E00", "#0072B2", "#009E73"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / "data.db")
    parser.add_argument("--classes", type=Path, default=ROOT / "s1_classified_genes.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/s1_classification_by_consensus_cluster.pdf")
    args = parser.parse_args()
    classifications = pd.read_csv(args.classes, sep="\t", usecols=["geneID", "S1_classification"])
    with sqlite3.connect(args.database) as connection:
        predictions = pd.read_sql_query(f'''SELECT gene_id, "{RUN_ID}" AS cluster
            FROM sequences WHERE label=1 AND gene_id IS NOT NULL AND "{RUN_ID}" IS NOT NULL''', connection)
    merged = predictions.merge(classifications, left_on="gene_id", right_on="geneID", how="inner")
    if merged.empty:
        raise ValueError("No classified S1 genes had an assigned consensus cluster")
    clusters = sorted(merged["cluster"].astype(int).unique())
    figure, axes = plt.subplots(1, len(clusters), figsize=(4.2 * len(clusters), 4.4), sharey=True, squeeze=False)
    for axis, cluster in zip(axes[0], clusters):
        counts = merged.loc[merged["cluster"] == cluster, "S1_classification"].value_counts().reindex(ORDER, fill_value=0)
        bars = axis.bar(range(len(ORDER)), counts.to_numpy(), color=COLOURS, width=0.72)
        axis.bar_label(bars, padding=3, fontsize=10)
        axis.set_xticks(range(len(ORDER)), ["Dependent", "Unaffected", "Repressed /\nacetyl-preferring"], rotation=30, ha="right")
        axis.set_title(f"Consensus cluster {cluster}\n(n={int(counts.sum())})")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0][0].set_ylabel("S1-classified gene count")
    figure.suptitle("Experimental S1 classification by predicted consensus cluster", y=1.02)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"matched_genes={len(merged)}, clusters={clusters}, saved={args.output}")


if __name__ == "__main__":
    main()
