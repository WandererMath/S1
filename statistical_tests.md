# S1 consensus-cluster enrichment statistics

## Inputs

- Consensus prediction column: `5-6-7-8` in `S1/data.db`
- Log2FC threshold: `0.58`
- Experimental classes: `results-58/genes.tsv`
- Tested matched genes: 3120
- Plotted classification memberships: 3120
- Full test table: `enrichment.tsv`

## Statistical test

For each consensus cluster and each S1 classification type, the analysis uses a one-sided Fisher exact test from SciPy (`scipy.stats.fisher_exact`, `alternative="greater"`). The 2x2 table is:

| | In this S1 class | In other S1 classes |
| --- | ---: | ---: |
| In this consensus cluster | observed count | cluster total - observed count |
| In other consensus clusters | class total - observed count | all other matched genes |

This test asks whether the S1 classification is enriched inside the cluster compared with all other assigned clusters. Fisher exact test is appropriate here because each test is a 2x2 categorical count comparison and remains valid for small or imbalanced cluster/class counts, which occur in this dataset. `PREFER_PTM`, `PREFER_NON_PTM`, and `NO_PREFER` are mutually exclusive labels based on the K247Q-versus-K247R TE log2FC. Only `PREFER_PTM` and `PREFER_NON_PTM` are tested; `NO_PREFER` is plotted but excluded from statistical testing.

Because multiple cluster-by-class combinations are tested, p-values are adjusted with the Benjamini-Hochberg false discovery rate procedure across all tests in this run. A cluster/class pair is treated as significantly enriched when `q_value_bh < 0.05` and `fold_enrichment > 1`.

## Plot annotation

The PDFs use horizontal stacked-bar layouts matching the tax result plots. `S1.pdf` shows each S1 class as a row with predicted consensus cluster count segments. `S1_fraction.pdf` uses the same rows but shows within-class cluster fractions. `cluster.pdf` shows each consensus cluster as a row with S1 class segments. Numbers inside count-plot bar segments are observed gene counts. An asterisk marks a class significantly enriched in that consensus cluster after Benjamini-Hochberg correction.

## Current result

Significant enriched cluster/class pairs at `q_value_bh < 0.05`: 1.

Smallest raw p-values in this run:

| Cluster | S1 classification | Observed | Fold enrichment | p-value | BH q-value |
| ---: | --- | ---: | ---: | ---: | ---: |
| 5 | PREFER_NON_PTM | 16 | 2.33 | 0.00106 | 0.0106 |
| 2 | PREFER_PTM | 132 | 1.1 | 0.0528 | 0.264 |
| 3 | PREFER_NON_PTM | 42 | 1.12 | 0.224 | 0.747 |
