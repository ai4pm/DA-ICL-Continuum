# Synthetic workflow

## Data-generating process

The ancestry plane contains centroids `(0.15, 0.20)`, `(0.85, 0.25)`, and `(0.50, 0.85)`. Of all samples, 65% are centroid-clustered with isotropic Gaussian standard deviation `0.055`; 35% are admixed. An admixed point selects one of the three centroid pairs, interpolates with `t ~ Beta(1.4, 1.4)`, and adds Gaussian standard deviation `0.035`. Coordinates are clipped to `[0, 1]`.

Two omics values are sampled independently from `Uniform(-1, 1)`. The base diagonal boundary has unit normal `(-1, 1)/sqrt(2)`. At ancestry coordinate `a`, the normal is rotated by `rotation * (a1 - 0.5)` and the boundary is shifted by `offset * (a2 - 0.5)`. The Bernoulli probability is the sigmoid of eight times the resulting signed margin.

The paper configuration uses 1,000 training rows, 600 test rows, KNN100, 30 replicates, and replicate seed derivation recorded in `metrics_by_replicate.csv`. The executable configuration is the numerical source of truth for the supplied synthetic images.

## Models

- **Discrete logistic:** assign training and query rows to the nearest centroid, then fit one standardized logistic regression per discrete group.
- **Empirical-continuum logistic:** for each query, fit standardized logistic regression on its 100 nearest training ancestry coordinates. If KNN100 contains only one outcome class, the neighborhood is expanded deterministically until both occur or all training rows are used.
- **Oracle-continuum logistic:** generate 100 labeled training rows at the query's exact ancestry coordinate from the known DGP and fit a local logistic model.
- **Oracle probability:** evaluate the known Bernoulli probability as a diagnostic upper reference.

## Outputs

- `config.resolved.json`: immutable resolved configuration.
- `metrics_by_replicate.csv`: AUROC, NLL, and Brier score by setting, replicate, and method.
- `summary_metrics.csv`: means, standard errors, and 95% normal intervals.
- `metrics_by_genetic_distance.csv`: replicate/bin metrics at rotation 10, offset 0.
- `summary_by_genetic_distance.csv`: aggregate distance curves.
- `figures/`: regenerated AUROC rotation/offset and AUROC/NLL distance figures.

The figure command consumes only the summary CSVs. It does not regenerate data or refit models.

