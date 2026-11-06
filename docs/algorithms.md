# Algorithm definitions

## Discrete ICL

For each query, `DiscreteICLClassifier` selects every development patient whose reported ancestry equals the query's `AFR` or `EUR` label. It uses no genetic KNN and no DACE. A query-specific TabPFN classifier conditions on that context's omics PCA100 rows and labels.

## Continuum ICL

`ContinuumICLClassifier` standardizes genotype PCA30 coordinates using development patients, computes squared Euclidean distances from a query to the combined AFR+EUR development pool, and selects the exact first `K=100` rows. Sorting uses `(distance, development_row_index)`, making exact ties deterministic. It then conditions TabPFN on the chosen rows' omics PCA100 features and labels.

## DA-ICL

`DAICLClassifier` starts from exactly the same KNN100 base as continuum ICL. It standardizes development AE20 coordinates, partitions the base and its complement by outcome class, and runs the following procedure separately for class 0 and class 1:

1. concatenate outside candidates before base-context rows;
2. calculate the complete Euclidean distance graph in AE20;
3. construct its minimum spanning tree with deterministic Prim traversal; and
4. add each outside candidate incident to an MST edge crossing into the base.

Labels only define the two candidate pools. Labels are never appended to AE coordinates or used as a numeric distance. Query AE values are unnecessary. The final context is `base + added`, with no duplicate development rows.

## TabPFN behavior

Each context is standardized in omics space, then used with TabPFN-2.6 (`n_estimators=32`, softmax temperature `0.9`). The Python dependency version and model checkpoint version are distinct: `tabpfn==7.1.0` loads the published TabPFN-2.6 checkpoint. If a final context contains one class, the implementation returns that observed class deterministically rather than invoking TabPFN.

`backend="logistic"` exists only to make API and unit tests fast on CPU. Results from that backend are not paper results.
