# S1 consensus-cluster enrichment statistics

## Inputs

- Consensus prediction column: `5-6-7-8` in `S1/data.db`
- Experimental classes: `S1/s1_classified_genes.tsv`
- Tested matched genes: 3120
- Full test table: `s1_classification_enrichment.tsv`

## Statistical test

For each consensus cluster and each S1 classification type, the analysis uses a one-sided Fisher exact test from SciPy (`scipy.stats.fisher_exact`, `alternative="greater"`). The 2x2 table is:

| | In this S1 class | In other S1 classes |
| --- | ---: | ---: |
| In this consensus cluster | observed count | cluster total - observed count |
| In other consensus clusters | class total - observed count | all other matched genes |

This test asks whether the S1 classification is enriched inside the cluster compared with all other assigned clusters. Fisher exact test is appropriate here because each test is a 2x2 categorical count comparison and remains valid for small or imbalanced cluster/class counts, which occur in this dataset.

Because multiple cluster-by-class combinations are tested, p-values are adjusted with the Benjamini-Hochberg false discovery rate procedure across all tests in this run. A cluster/class pair is treated as significantly enriched when `q_value_bh < 0.05` and `fold_enrichment > 1`.

## Plot annotation

The PDF uses a horizontal stacked-bar layout matching the tax result plots. Numbers inside bar segments are observed counts. An asterisk marks an S1 class that is significantly enriched in that consensus cluster after Benjamini-Hochberg correction.

## Current result

Significant enriched cluster/class pairs at `q_value_bh < 0.05`: 0.

Smallest raw p-values in this run:

| Cluster | S1 classification | Observed | Fold enrichment | p-value | BH q-value |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | S1_repressed_or_acetyl_preferring | 96 | 1.17 | 0.0125 | 0.187 |
| 4 | S1_unaffected | 370 | 1.02 | 0.0631 | 0.473 |
| 3 | S1_unaffected | 538 | 1.01 | 0.241 | 0.846 |
