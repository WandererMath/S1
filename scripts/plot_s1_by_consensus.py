#!/usr/bin/env python3
"""Plot S1 classes by consensus cluster and test class enrichment."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "5-6-7-8"
ORDER = ["S1_dependent", "S1_unaffected", "S1_repressed_or_acetyl_preferring"]
LABELS = {
    "S1_dependent": "S1 dependent",
    "S1_unaffected": "S1 unaffected",
    "S1_repressed_or_acetyl_preferring": "S1 repressed / acetyl-preferring",
}


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def load_merged(database: Path, classes_path: Path, run_id: str) -> pd.DataFrame:
    classifications = pd.read_csv(classes_path, sep="\t", usecols=["geneID", "S1_classification"])
    consensus_column = quote_identifier(run_id)
    with sqlite3.connect(database) as connection:
        predictions = pd.read_sql_query(f"""SELECT gene_id, {consensus_column} AS cluster
            FROM sequences
            WHERE label=1 AND gene_id IS NOT NULL AND {consensus_column} IS NOT NULL""", connection)
    merged = predictions.merge(classifications, left_on="gene_id", right_on="geneID", how="inner")
    if merged.empty:
        raise ValueError("No classified S1 genes had an assigned consensus cluster")
    merged["cluster"] = merged["cluster"].astype(int)
    return merged


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    adjusted = [1.0] * len(p_values)
    ranked = sorted(enumerate(p_values), key=lambda item: item[1], reverse=True)
    running = 1.0
    total = len(p_values)
    for rank_from_largest, (index, p_value) in enumerate(ranked):
        rank = total - rank_from_largest
        running = min(running, p_value * total / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted


def enrichment_table(merged: pd.DataFrame, clusters: list[int]) -> pd.DataFrame:
    rows = []
    total = len(merged)
    for cluster in clusters:
        in_cluster = merged["cluster"] == cluster
        cluster_total = int(in_cluster.sum())
        for classification in ORDER:
            in_class = merged["S1_classification"] == classification
            class_total = int(in_class.sum())
            observed = int((in_cluster & in_class).sum())
            table = [
                [observed, cluster_total - observed],
                [class_total - observed, total - cluster_total - class_total + observed],
            ]
            odds_ratio, p_value = fisher_exact(table, alternative="greater")
            expected = cluster_total * class_total / total
            rows.append({
                "cluster": cluster,
                "s1_classification": classification,
                "cluster_total": cluster_total,
                "class_total": class_total,
                "observed": observed,
                "expected": expected,
                "fold_enrichment": observed / expected if expected else np.nan,
                "odds_ratio": odds_ratio,
                "p_value": p_value,
            })
    table = pd.DataFrame(rows)
    table["q_value_bh"] = benjamini_hochberg(table["p_value"].tolist())
    return table


def plot_counts(merged: pd.DataFrame, enrichment: pd.DataFrame, output_path: Path) -> None:
    counts = (
        merged.groupby(["cluster", "S1_classification"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ORDER, fill_value=0)
        .sort_index()
    )
    clusters = counts.index.tolist()
    colors = plt.get_cmap("tab20")(np.arange(len(ORDER)))
    class_colors = dict(zip(ORDER, colors))
    significant = {
        (int(row.cluster), row.s1_classification)
        for row in enrichment.itertuples()
        if row.q_value_bh < 0.05 and row.fold_enrichment > 1
    }

    y = np.arange(len(clusters))
    fig, axis = plt.subplots(figsize=(14, max(5, len(clusters) * 0.62 + 2)))
    max_total = int(counts.sum(axis=1).max())
    for index, cluster in enumerate(clusters):
        left = 0
        row = counts.loc[cluster]
        for classification, count in sorted(
            row.items(),
            key=lambda item: (-item[1], ORDER.index(item[0])),
        ):
            if count == 0:
                continue
            axis.barh(index, count, left=left, height=0.8, color=class_colors[classification])
            if count >= max(8, max_total * 0.035):
                marker = "*" if (int(cluster), classification) in significant else ""
                axis.text(
                    left + count / 2,
                    index,
                    f"{int(count)}{marker}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold" if marker else "normal",
                )
            left += int(count)
        axis.text(left + max_total * 0.012, index, str(int(row.sum())), va="center", fontsize=9)

    axis.set_yticks(y)
    axis.set_yticklabels([f"Consensus cluster {cluster}" for cluster in clusters])
    axis.invert_yaxis()
    axis.set_xlabel("S1-classified gene count")
    axis.set_ylabel("Predicted consensus cluster")
    axis.set_title("Experimental S1 classification counts by predicted consensus cluster")
    axis.legend(
        handles=[Patch(color=class_colors[classification], label=LABELS[classification]) for classification in ORDER],
        title="S1 classification",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    fig.text(0.01, 0.01, "* Fisher exact enrichment, BH FDR q < 0.05", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 0.8, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_methods(path: Path, run_id: str, matched_genes: int, enrichment_path: Path, enrichment: pd.DataFrame) -> None:
    significant = enrichment[(enrichment["q_value_bh"] < 0.05) & (enrichment["fold_enrichment"] > 1)]
    top = enrichment.sort_values("p_value").head(3)
    top_rows = "\n".join(
        f"| {int(row.cluster)} | {row.s1_classification} | {row.observed} | {row.fold_enrichment:.3g} | {row.p_value:.3g} | {row.q_value_bh:.3g} |"
        for row in top.itertuples()
    )
    path.write_text(
        f"""# S1 consensus-cluster enrichment statistics

## Inputs

- Consensus prediction column: `{run_id}` in `S1/data.db`
- Experimental classes: `S1/s1_classified_genes.tsv`
- Tested matched genes: {matched_genes}
- Full test table: `{enrichment_path.name}`

## Statistical test

For each consensus cluster and each S1 classification type, the analysis uses a one-sided Fisher exact test from SciPy (`scipy.stats.fisher_exact`, `alternative=\"greater\"`). The 2x2 table is:

| | In this S1 class | In other S1 classes |
| --- | ---: | ---: |
| In this consensus cluster | observed count | cluster total - observed count |
| In other consensus clusters | class total - observed count | all other matched genes |

This test asks whether the S1 classification is enriched inside the cluster compared with all other assigned clusters. Fisher exact test is appropriate here because each test is a 2x2 categorical count comparison and remains valid for small or imbalanced cluster/class counts, which occur in this dataset.

Because multiple cluster-by-class combinations are tested, p-values are adjusted with the Benjamini-Hochberg false discovery rate procedure across all tests in this run. A cluster/class pair is treated as significantly enriched when `q_value_bh < 0.05` and `fold_enrichment > 1`.

## Plot annotation

The PDF uses a horizontal stacked-bar layout matching the tax result plots. Numbers inside bar segments are observed counts. An asterisk marks an S1 class that is significantly enriched in that consensus cluster after Benjamini-Hochberg correction.

## Current result

Significant enriched cluster/class pairs at `q_value_bh < 0.05`: {len(significant)}.

Smallest raw p-values in this run:

| Cluster | S1 classification | Observed | Fold enrichment | p-value | BH q-value |
| ---: | --- | ---: | ---: | ---: | ---: |
{top_rows}
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--database", type=Path, default=ROOT / "data.db")
    parser.add_argument("--classes", type=Path, default=ROOT / "s1_classified_genes.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / f"results/{RUN_ID}/s1_classification_by_consensus_cluster.pdf")
    parser.add_argument("--enrichment-output", type=Path, default=ROOT / f"results/{RUN_ID}/s1_classification_enrichment.tsv")
    parser.add_argument("--methods-output", type=Path, default=ROOT / f"results/{RUN_ID}/statistical_tests.md")
    args = parser.parse_args()
    merged = load_merged(args.database, args.classes, args.run_id)
    clusters = [int(cluster) for cluster in sorted(merged["cluster"].unique())]
    enrichment = enrichment_table(merged, clusters)
    args.enrichment_output.parent.mkdir(parents=True, exist_ok=True)
    enrichment.to_csv(args.enrichment_output, sep="\t", index=False)
    plot_counts(merged, enrichment, args.output)
    args.methods_output.parent.mkdir(parents=True, exist_ok=True)
    write_methods(args.methods_output, args.run_id, len(merged), args.enrichment_output, enrichment)
    significant_count = int(((enrichment["q_value_bh"] < 0.05) & (enrichment["fold_enrichment"] > 1)).sum())
    print(
        f"matched_genes={len(merged)}, clusters={clusters}, "
        f"significant_enrichments={significant_count}, saved={args.output}"
    )


if __name__ == "__main__":
    main()
