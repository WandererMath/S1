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
DEFAULT_THRESHOLD = 0.58
ORDER = ["S1_dependent", "Acetylated-S1-Favored", "S1_unaffected"]
ACTIVE_LABELS = ["S1_dependent", "Acetylated-S1-Favored"]
DISPLAY_ORDER = [
    "S1_dependent_only",
    "Acetylated-S1-Favored_only",
    "S1_dependent_and_Acetylated-S1-Favored",
    "S1_unaffected",
]
LABELS = {
    "S1_dependent": "S1 dependent",
    "Acetylated-S1-Favored": "Acetylated-S1-Favored",
    "S1_unaffected": "S1 unaffected",
    "S1_dependent_only": "S1 dependent only",
    "Acetylated-S1-Favored_only": "Acetylated-S1-Favored only",
    "S1_dependent_and_Acetylated-S1-Favored": "Both",
}
DISPLAY_MEMBERSHIPS = {
    "S1_dependent_only": ["S1_dependent"],
    "Acetylated-S1-Favored_only": ["Acetylated-S1-Favored"],
    "S1_dependent_and_Acetylated-S1-Favored": ["S1_dependent", "Acetylated-S1-Favored"],
    "S1_unaffected": ["S1_unaffected"],
}


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def threshold_tag(threshold: float) -> str:
    return str(int(round(threshold * 100)))


def threshold_results_dir(threshold: float) -> Path:
    return ROOT / f"results-{threshold_tag(threshold)}"


def parse_classification(value: str) -> list[str]:
    labels = [label.strip() for label in str(value).split(";") if label.strip()]
    active = [label for label in ACTIVE_LABELS if label in labels]
    if active:
        return active
    return ["S1_unaffected"]


def display_classification(labels: list[str]) -> str:
    label_set = set(labels)
    if all(label in label_set for label in ACTIVE_LABELS):
        return "S1_dependent_and_Acetylated-S1-Favored"
    if "S1_dependent" in label_set:
        return "S1_dependent_only"
    if "Acetylated-S1-Favored" in label_set:
        return "Acetylated-S1-Favored_only"
    return "S1_unaffected"


def load_merged(database: Path, classes_path: Path, run_id: str) -> pd.DataFrame:
    classifications = pd.read_csv(classes_path, sep="\t", usecols=["geneID", "S1_classification"])
    classifications["S1_labels"] = classifications["S1_classification"].map(parse_classification)
    classifications["S1_display_classification"] = classifications["S1_labels"].map(display_classification)
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
    total = merged["gene_id"].nunique()
    for cluster in clusters:
        cluster_genes = set(merged.loc[merged["cluster"] == cluster, "gene_id"])
        cluster_total = len(cluster_genes)
        for classification in ORDER:
            has_class = merged["S1_labels"].map(lambda labels: classification in labels)
            class_genes = set(merged.loc[has_class, "gene_id"])
            class_total = len(class_genes)
            observed = len(cluster_genes & class_genes)
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


def significant_pairs(enrichment: pd.DataFrame) -> set[tuple[int, str]]:
    return {
        (int(row.cluster), row.s1_classification)
        for row in enrichment.itertuples()
        if row.q_value_bh < 0.05 and row.fold_enrichment > 1
    }


def plot_s1_counts(merged: pd.DataFrame, enrichment: pd.DataFrame, output_path: Path) -> None:
    counts = (
        merged.groupby(["S1_display_classification", "cluster"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=DISPLAY_ORDER, fill_value=0)
    )
    clusters = sorted(counts.columns.tolist())
    colors = plt.get_cmap("tab20")(np.arange(len(clusters)))
    cluster_colors = dict(zip(clusters, colors))
    significant = significant_pairs(enrichment)

    y = np.arange(len(DISPLAY_ORDER))
    fig, axis = plt.subplots(figsize=(14, max(5, len(DISPLAY_ORDER) * 0.62 + 2)))
    max_total = int(counts.sum(axis=1).max())
    for index, classification in enumerate(DISPLAY_ORDER):
        left = 0
        row = counts.loc[classification]
        for cluster, count in sorted(
            row.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            if count == 0:
                continue
            axis.barh(index, count, left=left, height=0.8, color=cluster_colors[cluster])
            if count >= max(8, max_total * 0.035):
                marker = "*" if any((int(cluster), label) in significant for label in DISPLAY_MEMBERSHIPS[classification]) else ""
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
    axis.set_yticklabels([LABELS[classification] for classification in DISPLAY_ORDER])
    axis.invert_yaxis()
    axis.set_xlabel("Gene count")
    axis.set_ylabel("S1 classification")
    axis.set_title("Predicted consensus cluster counts by S1 classification")
    axis.legend(
        handles=[Patch(color=cluster_colors[cluster], label=f"Cluster {cluster}") for cluster in clusters],
        title="Predicted consensus cluster",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_s1_fractions(merged: pd.DataFrame, output_path: Path) -> None:
    counts = (
        merged.groupby(["S1_display_classification", "cluster"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=DISPLAY_ORDER, fill_value=0)
    )
    fractions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    clusters = sorted(fractions.columns.tolist())
    colors = plt.get_cmap("tab20")(np.arange(len(clusters)))
    cluster_colors = dict(zip(clusters, colors))

    y = np.arange(len(DISPLAY_ORDER))
    fig, axis = plt.subplots(figsize=(14, max(5, len(DISPLAY_ORDER) * 0.62 + 2)))
    for index, classification in enumerate(DISPLAY_ORDER):
        left = 0.0
        row = fractions.loc[classification]
        for cluster, fraction in sorted(
            row.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            if fraction == 0:
                continue
            axis.barh(index, fraction, left=left, height=0.8, color=cluster_colors[cluster])
            if fraction >= 0.06:
                axis.text(
                    left + fraction / 2,
                    index,
                    f"{fraction:.0%}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
            left += float(fraction)

    axis.set_yticks(y)
    axis.set_yticklabels([LABELS[classification] for classification in DISPLAY_ORDER])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Fraction of genes")
    axis.set_ylabel("S1 classification")
    axis.set_title("Predicted consensus cluster fractions by S1 classification")
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.legend(
        handles=[Patch(color=cluster_colors[cluster], label=f"Cluster {cluster}") for cluster in clusters],
        title="Predicted consensus cluster",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_counts(merged: pd.DataFrame, enrichment: pd.DataFrame, output_path: Path) -> None:
    counts = (
        merged.groupby(["cluster", "S1_display_classification"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=DISPLAY_ORDER, fill_value=0)
        .sort_index()
    )
    clusters = counts.index.tolist()
    colors = plt.get_cmap("tab20")(np.arange(len(DISPLAY_ORDER)))
    class_colors = dict(zip(DISPLAY_ORDER, colors))
    significant = significant_pairs(enrichment)

    y = np.arange(len(clusters))
    fig, axis = plt.subplots(figsize=(14, max(5, len(clusters) * 0.62 + 2)))
    max_total = int(counts.sum(axis=1).max())
    for index, cluster in enumerate(clusters):
        left = 0
        row = counts.loc[cluster]
        for classification, count in sorted(
            row.items(),
            key=lambda item: (-item[1], DISPLAY_ORDER.index(item[0])),
        ):
            if count == 0:
                continue
            axis.barh(index, count, left=left, height=0.8, color=class_colors[classification])
            if count >= max(8, max_total * 0.035):
                marker = "*" if any((int(cluster), label) in significant for label in DISPLAY_MEMBERSHIPS[classification]) else ""
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
    axis.set_xlabel("Gene count")
    axis.set_ylabel("Predicted consensus cluster")
    axis.set_title("S1 classification counts by predicted consensus cluster")
    axis.legend(
        handles=[Patch(color=class_colors[classification], label=LABELS[classification]) for classification in DISPLAY_ORDER],
        title="S1 classification",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_methods(path: Path, run_id: str, threshold: float, classes_path: Path, matched_genes: int, membership_count: int, overlap_count: int, enrichment_path: Path, enrichment: pd.DataFrame) -> None:
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
- Log2FC threshold: `{threshold:g}`
- Experimental classes: `{classes_path.relative_to(ROOT)}`
- Tested matched genes: {matched_genes}
- Plotted classification memberships: {membership_count}
- Genes assigned both `S1_dependent` and `Acetylated-S1-Favored`: {overlap_count}
- Full test table: `{enrichment_path.name}`

## Statistical test

For each consensus cluster and each S1 classification type, the analysis uses a one-sided Fisher exact test from SciPy (`scipy.stats.fisher_exact`, `alternative=\"greater\"`). The 2x2 table is:

| | In this S1 class | In other S1 classes |
| --- | ---: | ---: |
| In this consensus cluster | observed count | cluster total - observed count |
| In other consensus clusters | class total - observed count | all other matched genes |

This test asks whether the S1 classification is enriched inside the cluster compared with all other assigned clusters. Fisher exact test is appropriate here because each test is a 2x2 categorical count comparison and remains valid for small or imbalanced cluster/class counts, which occur in this dataset. `S1_dependent` and `Acetylated-S1-Favored` are tested as independent labels, while `S1_unaffected` is assigned only to genes with neither active label.

Because multiple cluster-by-class combinations are tested, p-values are adjusted with the Benjamini-Hochberg false discovery rate procedure across all tests in this run. A cluster/class pair is treated as significantly enriched when `q_value_bh < 0.05` and `fold_enrichment > 1`.

## Plot annotation

The PDFs use horizontal stacked-bar layouts matching the tax result plots. For plotting, labels are converted to mutually exclusive display bins: `S1 dependent only`, `Acetylated-S1-Favored only`, `Both`, and `S1 unaffected`. `S1.pdf` shows each S1 display class as a row with predicted consensus cluster count segments. `S1_fraction.pdf` uses the same rows but shows within-class cluster fractions. `cluster.pdf` shows each consensus cluster as a row with S1 display-class segments. This makes genes carrying both active labels visible without also counting them as unaffected. Numbers inside count-plot bar segments are observed gene counts. An asterisk marks a display bin whose underlying independent label is significantly enriched in that consensus cluster after Benjamini-Hochberg correction.

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
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--database", type=Path, default=ROOT / "data.db")
    parser.add_argument("--classes", type=Path)
    parser.add_argument("--s1-output", type=Path)
    parser.add_argument("--s1-fraction-output", type=Path)
    parser.add_argument("--cluster-output", type=Path)
    parser.add_argument("--enrichment-output", type=Path)
    parser.add_argument("--methods-output", type=Path)
    args = parser.parse_args()
    run_dir = threshold_results_dir(args.threshold) / args.run_id
    classes_path = args.classes or threshold_results_dir(args.threshold) / "genes.tsv"
    s1_output = args.s1_output or run_dir / "S1.pdf"
    s1_fraction_output = args.s1_fraction_output or run_dir / "S1_fraction.pdf"
    cluster_output = args.cluster_output or run_dir / "cluster.pdf"
    enrichment_output = args.enrichment_output or run_dir / "s1_classification_enrichment.tsv"
    methods_output = args.methods_output or ROOT / "statistical_tests.md"
    merged = load_merged(args.database, classes_path, args.run_id)
    clusters = [int(cluster) for cluster in sorted(merged["cluster"].unique())]
    enrichment = enrichment_table(merged, clusters)
    enrichment_output.parent.mkdir(parents=True, exist_ok=True)
    enrichment.to_csv(enrichment_output, sep="\t", index=False)
    plot_s1_counts(merged, enrichment, s1_output)
    plot_s1_fractions(merged, s1_fraction_output)
    plot_cluster_counts(merged, enrichment, cluster_output)
    methods_output.parent.mkdir(parents=True, exist_ok=True)
    matched_genes = merged["gene_id"].nunique()
    membership_count = int(merged["S1_labels"].map(len).sum())
    overlap_count = int(merged["S1_display_classification"].eq("S1_dependent_and_Acetylated-S1-Favored").sum())
    write_methods(methods_output, args.run_id, args.threshold, classes_path, matched_genes, membership_count, overlap_count, enrichment_output, enrichment)
    significant_count = int(((enrichment["q_value_bh"] < 0.05) & (enrichment["fold_enrichment"] > 1)).sum())
    print(
        f"matched_genes={matched_genes}, classification_memberships={membership_count}, overlaps={overlap_count}, clusters={clusters}, "
        f"significant_enrichments={significant_count}, saved={s1_output},{s1_fraction_output},{cluster_output}"
    )


if __name__ == "__main__":
    main()
