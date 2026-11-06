# Prepared-array contract

## Purpose

The estimators follow the ordinary scikit-learn pattern: each row is a patient, each column is a feature, and `fit(X, y, ...)` receives a development pool. There is no repository-specific patient table or multi-task file schema.

## Required preparation outside this repository

The caller supplies:

1. omics PCA100 prediction matrices;
2. genotype PCA30 coordinates;
3. omics AE20 embeddings for development patients when using DACE;
4. binary development labels;
5. reported `AFR`/`EUR` labels for the discrete method; and
6. a leakage-free development/test partition.

All imputers, scalers, PCA models, and autoencoders must be fitted according to the caller's approved protocol. Test outcomes must not affect those transformations or context construction. The public estimators standardize the already prepared genotype coordinates over the development pool for KNN search, standardize AE20 over the development pool for DACE, and standardize omics features within each final context before TabPFN prediction.

The API accepts NumPy arrays and pandas DataFrames. DataFrame indices are not used to align blocks: rows must already be in the same order. Supply `sample_ids` to preserve auditable identifiers in `last_context_report_`.

## Data access

TCGA open-access molecular data can be obtained from the [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/). TCGA controlled-access germline genotypes require dbGaP authorization. Investigators must apply for the relevant study through dbGaP, obtain institutional approval when required, and process the files only in an approved secure environment. This repository neither redistributes nor automates retrieval of controlled-access data.

