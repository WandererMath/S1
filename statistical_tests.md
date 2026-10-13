# S1 consensus-cluster enrichment statistics

## Inputs

- Consensus prediction column: `5-6-7-8` in `S1/data.db`
- Log2FC threshold: `0.58`
- Experimental classes: `results-58/genes.tsv`
- Tested matched genes: 3120
- Plotted classification memberships: 3125
- Genes assigned both `S1_dependent` and `Acetylated-S1-Favored`: 5
- Full test table: `s1_classification_enrichment.tsv`

## Statistical test

For each consensus cluster and each S1 classification type, the analysis uses a one-sided Fisher exact test from SciPy (`scipy.stats.fisher_exact`, `alternative="greater"`). The 2x2 table is:

| | In this S1 class | In other S1 classes |
| --- | ---: | ---: |
| In this consensus cluster | observed count | cluster total - observed count |
| In other consensus clusters | class total - observed count | all other matched genes |

This test asks whether the S1 classification is enriched inside the cluster compared with all other assigned clusters. Fisher exact test is appropriate here because each test is a 2x2 categorical count comparison and remains valid for small or imbalanced cluster/class counts, which occur in this dataset. `S1_dependent` and `Acetylated-S1-Favored` are tested as independent labels, while `S1_unaffected` is assigned only to genes with neither active label.

Because multiple cluster-by-class combinations are tested, p-values are adjusted with the Benjamini-Hochberg false discovery rate procedure across all tests in this run. A cluster/class pair is treated as significantly enriched when `q_value_bh < 0.05` and `fold_enrichment > 1`.

## Plot annotation

The PDF uses a horizontal stacked-bar layout matching the tax result plots. For plotting, labels are converted to mutually exclusive display bins: `S1 dependent only`, `Acetylated-S1-Favored only`, `Both`, and `S1 unaffected`. This makes genes carrying both active labels visible without also counting them as unaffected. Numbers inside bar segments are observed gene counts. An asterisk marks a display bin whose underlying independent label is significantly enriched in that consensus cluster after Benjamini-Hochberg correction.

## Current result

Significant enriched cluster/class pairs at `q_value_bh < 0.05`: 0.

Smallest raw p-values in this run:

| Cluster | S1 classification | Observed | Fold enrichment | p-value | BH q-value |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | Acetylated-S1-Favored | 132 | 1.1 | 0.0528 | 0.387 |
| 5 | S1_dependent | 11 | 1.61 | 0.0763 | 0.387 |
| 1 | S1_unaffected | 338 | 1.03 | 0.0775 | 0.387 |
