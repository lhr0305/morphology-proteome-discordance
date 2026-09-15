# Morphology–proteome discordance in colon and pancreatic cancer

Analysis code for the study *"Morphology–Proteome Discordance Maps Proteomic Structure Unresolved by Histology in Colon and Pancreatic Cancer."*

The pipeline quantifies how much pathway-level proteomic and phosphoproteomic state can be
predicted from routine H&E whole-slide images, defines **morphology–proteome discordance** as
the strictly out-of-fold residual, resolves its recurrent programs, tests them head-to-head
against the observed proteome, traces them with matched and external transcriptomes, and
localizes them back within whole-slide images.

## Data

No patient data are redistributed here. All inputs are public:

| Source | Content | Access |
|---|---|---|
| NCI Proteomic Data Commons | CPTAC COAD / PDAC proteome and phosphoproteome quantitation | https://pdc.cancer.gov |
| NCI Imaging Data Commons | CPTAC whole-slide images (DICOM) | https://imaging.datacommons.cancer.gov |
| NCI Genomic Data Commons | TCGA COAD / PAAD / STAD transcriptomes, slides, clinical fields | https://portal.gdc.cancer.gov |

Discovery is restricted to primary tumors with matched slide + proteome + phosphoproteome
(final set: 236 patients, 97 COAD and 139 PDAC). TCGA cohorts are used only as external
transcriptomic and image-transfer extensions.

## Pipeline

Stages run in order; each writes its outputs to `--output-dir` and each later stage reads the
prior stage's directory. Every script exposes `--help`.

```
stage1_finalize_download_registry.py     # freeze the download registry / file provenance
stage2_build_master_table.py             # per-patient master table: omics, slides, clinical endpoints
stage3_build_omics_matrices.py           # protein + phosphosite matrices -> pathway / kinase-PTM scores
stage3_build_exact_phospho_activity.py   # site-specific kinase and PTM activity scores
stage4_prepare_dicom_tiles.py            # slide QC, representative-slide selection, tile coordinates
stage4_extract_embeddings.py             # frozen pathology-encoder tile embeddings
stage4_aggregate_multislide_embeddings.py# patient-level multi-slide aggregation
stage5_train_morphology_omics.py         # nested patient-level CV histology->omics prediction (OOF only)
stage5_full_retrain_permutation_null.py  # full-retraining permutation null
stage6_build_discordance.py              # standardized observed-minus-OOF residual matrices
stage7_discover_archetypes.py            # consensus NMF: rank selection, loadings, cross-cohort transfer
stage8_clinical_hidden_aggression.py     # clinical association models and meta-analysis
stage9_spatial_localization.py           # attention-based tile scoring, motif catalogue
stage9_hovernet_quantification.py        # nuclei segmentation / classification on gated tiles
stage10_prepare_tcga_wsi_manifests.py    # representative TCGA slide manifests
stage10_prepare_svs_tiles.py             # TCGA SVS tile coordinates
stage10_extract_svs_embeddings.py        # TCGA tile embeddings
stage10_project_tcga_wsi_archetypes.py   # project discovery loadings onto TCGA slides
stage10_external_validation.py           # TCGA RNA-proxy external extension + random-signature null
stage11_robustness_benchmark.py          # technical repeat, encoder/aggregation and artifact checks
```

Example:

```bash
python scripts/stage5_train_morphology_omics.py \
    --project-root /path/to/project \
    --output-dir results/stage5 \
    --folds 5 --repeats 5 --permutations 100 --seed 20260609

python scripts/stage6_build_discordance.py \
    --project-root /path/to/project \
    --prediction-dir results/stage5 \
    --output-dir results/stage6
```

## Reproducibility

- All preprocessing (dimensionality reduction, scaling, feature filtering) is fit inside
  training folds only, and all slides from a patient stay in one fold.
- Only out-of-fold predictions are retained, so the discordance residual never sees a
  model that was fit on the same patient.
- Random seeds are explicit CLI arguments; the seed used throughout the paper is `20260609`.
- Prediction significance is assessed both against a post-hoc label-shuffle null and against
  a full-retraining permutation null in which the whole pipeline is refit on permuted labels.

## Requirements

Python 3.9+. See `requirements.txt`. The pathology encoders (Prov-GigaPath, H-optimus-0) and
the nuclei model (HoVer-Net) are pulled from their public model repositories at runtime and
are subject to their own licences.

## Citation

Li S, Wang Y: Morphology–proteome discordance maps proteomic structure unresolved by
histology in colon and pancreatic cancer. *Cancer Genomics & Proteomics*, submitted.

## Licence

MIT (see `LICENSE`).
