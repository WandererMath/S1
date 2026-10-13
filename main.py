from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_THRESHOLD = 0.58


def threshold_tag(threshold: float) -> str:
    return str(int(round(threshold * 100)))


def output_directory(threshold: float) -> Path:
    return ROOT / f"results-{threshold_tag(threshold)}"


def classify_s1(row: pd.Series, threshold: float) -> str:
    labels = []
    if row["log2FC_TE_K247Q_vs_WT"] <= -threshold:
        labels.append("S1_dependent")
    if row["log2FC_TE_K247Q_vs_K247R"] >= threshold:
        labels.append("Acetylated-S1-Favored")
    if not labels:
        labels.append("S1_unaffected")
    return ";".join(labels)


def build_classifications(raw_path: Path, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(raw_path, sep="\t")

    df["RNA_WT_mean"] = df[["RNA_WT-1", "RNA_WT-2", "RNA_WT-3"]].mean(axis=1)
    df["RNA_K247Q_mean"] = df[["RNA_K247Q-1", "RNA_K247Q-2", "RNA_K247Q-3"]].mean(axis=1)
    df["RNA_K247R_mean"] = df[["RNA_K247R-1", "RNA_K247R-2", "RNA_K247R-3"]].mean(axis=1)

    df["RNC_WT_mean"] = df[["RNC_WT-1", "RNC_WT-2", "RNC_WT-3"]].mean(axis=1)
    df["RNC_K247Q_mean"] = df[["RNC_K247Q-1", "RNC_K247Q-2", "RNC_K247Q-3"]].mean(axis=1)
    df["RNC_K247R_mean"] = df[["RNC_K247R-1", "RNC_K247R-2", "RNC_K247R-3"]].mean(axis=1)

    epsilon = 1e-3
    df["TE_WT"] = (df["RNC_WT_mean"] + epsilon) / (df["RNA_WT_mean"] + epsilon)
    df["TE_K247Q"] = (df["RNC_K247Q_mean"] + epsilon) / (df["RNA_K247Q_mean"] + epsilon)
    df["TE_K247R"] = (df["RNC_K247R_mean"] + epsilon) / (df["RNA_K247R_mean"] + epsilon)

    df["log2FC_TE_K247Q_vs_WT"] = np.log2(df["TE_K247Q"] / df["TE_WT"])
    df["log2FC_TE_K247R_vs_WT"] = np.log2(df["TE_K247R"] / df["TE_WT"])
    df["log2FC_TE_K247Q_vs_K247R"] = np.log2(df["TE_K247Q"] / df["TE_K247R"])
    df["S1_classification"] = df.apply(classify_s1, axis=1, threshold=threshold)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify S1 initiation labels from raw RNC/RNA data.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--raw", type=Path, default=ROOT / "raw.tsv")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or output_directory(args.threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = build_classifications(args.raw, args.threshold)

    output_cols = [
        "geneID",
        "TE_WT",
        "TE_K247Q",
        "TE_K247R",
        "log2FC_TE_K247Q_vs_WT",
        "log2FC_TE_K247R_vs_WT",
        "log2FC_TE_K247Q_vs_K247R",
        "S1_classification",
    ]
    output_path = output_dir / "genes.tsv"
    df[output_cols].to_csv(output_path, sep="\t", index=False)
    print(f"threshold={args.threshold:g}, genes={len(df)}, saved={output_path}")


if __name__ == "__main__":
    main()