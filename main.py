import numpy as np
import pandas as pd

# 1. Load the data
df = pd.read_csv("raw.tsv", sep="\t")

# 2. Compute replicate means for each assay and condition
df["RNA_WT_mean"] = df[["RNA_WT-1", "RNA_WT-2", "RNA_WT-3"]].mean(axis=1)
df["RNA_K247Q_mean"] = df[
    ["RNA_K247Q-1", "RNA_K247Q-2", "RNA_K247Q-3"]
].mean(axis=1)
df["RNA_K247R_mean"] = df[
    ["RNA_K247R-1", "RNA_K247R-2", "RNA_K247R-3"]
].mean(axis=1)

df["RNC_WT_mean"] = df[["RNC_WT-1", "RNC_WT-2", "RNC_WT-3"]].mean(axis=1)
df["RNC_K247Q_mean"] = df[
    ["RNC_K247Q-1", "RNC_K247Q-2", "RNC_K247Q-3"]
].mean(axis=1)
df["RNC_K247R_mean"] = df[
    ["RNC_K247R-1", "RNC_K247R-2", "RNC_K247R-3"]
].mean(axis=1)

# Add a small pseudo-count to prevent division by zero for lowly expressed genes
epsilon = 1e-3

# 3. Calculate Translation Efficiency (TE)
df["TE_WT"] = (df["RNC_WT_mean"] + epsilon) / (df["RNA_WT_mean"] + epsilon)
df["TE_K247Q"] = (df["RNC_K247Q_mean"] + epsilon) / (
    df["RNA_K247Q_mean"] + epsilon
)
df["TE_K247R"] = (df["RNC_K247R_mean"] + epsilon) / (
    df["RNA_K247R_mean"] + epsilon
)

# 4. Calculate log2 fold changes of TE
df["log2FC_TE_K247Q_vs_WT"] = np.log2(df["TE_K247Q"] / df["TE_WT"])
df["log2FC_TE_K247R_vs_WT"] = np.log2(df["TE_K247R"] / df["TE_WT"])
df["log2FC_TE_K247Q_vs_K247R"] = np.log2(df["TE_K247Q"] / df["TE_K247R"])

# 5. Classify S1 initiation profile as independent labels.
def classify_s1(row):
    labels = []
    if row["log2FC_TE_K247Q_vs_WT"] <= -1.0:
        labels.append("S1_dependent")
    if row["log2FC_TE_K247Q_vs_K247R"] >= 1.0:
        labels.append("Acetylated-S1-Favored")
    if not labels:
        labels.append("S1_unaffected")
    return ";".join(labels)


df["S1_classification"] = df.apply(classify_s1, axis=1)

# 6. Save results
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
df[output_cols].to_csv("s1_classified_genes.tsv", sep="\t", index=False)