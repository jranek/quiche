# QUICHE

QUantitative InterCellular nicHe Enrichment

## Introduction

QUICHE is a statistical differential abundance testing method that can be used to discover cellular niches differentially enriched in spatial regions, longitudinal samples, or clinical patient groups. For more details on the method, please read the associated preprint: [Ranek JS, Greenwald NF, Goldston M, Camacho Fullaway C, Sowers C, Kong A, Mouron S, Quintela-Fandino M, West RB, Angelo M. QUICHE reveals structural definitions of anti-tumor responses in triple negative breast cancer. 2024](https://www.biorxiv.org/content/10.1101/2025.01.06.631548v1).

<p>
  <img src="https://github.com/jranek/quiche/blob/main/docs/pipeline.png?raw=True" />
</p>

This repo is currently under development as we are in the process of porting over our existing code into this independent repository. In the meantime, you can access the code associated with the paper [here](https://github.com/angelolab/publications/tree/main/2024-Ranek_etal_QUICHE). 

## Modernization summary (2026-02)
The codebase has been modernized for current Python and packaging standards while preserving the public API (`quiche.pp`, `quiche.tl`, `quiche.pl`).

- Python support updated to `3.10-3.12`.
- Packaging metadata migrated to modern PEP 621 format in `pyproject.toml`.
- Dependencies split into install extras:
  - `.[plot]` for plotting dependencies
  - `.[full]` for heavier spatial/omics workflows
  - `.[dev]` for tests and development tooling
- Optional heavy dependencies now fail with clear messages only when those specific workflows are used, instead of breaking base imports.
- Compatibility fixes applied for older grammar-era patterns and runtime issues.
- Automated tests and CI were added (GitHub Actions matrix for Python 3.10/3.11/3.12).

For full migration details, see `docs/migration-python-modernization.md`.

## Data access
You can download all of the preprocessed MIBI-TOF datasets (`.h5ad` files) from the [Zenodo](https://zenodo.org/records/14290163) repository. Imaging data and cell segmentation masks can be found in the [BioStudies](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1507) repository. 

## Installation
Supported Python versions: **3.10, 3.11, 3.12**.

You can clone the git repository by,
```
git clone https://github.com/jranek/quiche.git
```
Then change the working directory as,
```
cd quiche
```

### Option 1: pip (recommended for Python-only workflows)
Install core dependencies:
```bash
pip install .
```

Install optional extras:
```bash
# plotting helpers
pip install .[plot]

# full spatial/omics stack (scanpy/squidpy/pertpy/muon/sketchKH)
pip install .[full]

# development + test tooling
pip install .[dev]
```

### Option 2: conda (recommended for R/edgeR workflows)
Create the conda environment using the provided YAML file:

```
conda env create -f venv_quiche.yml
```

Once the environment is created, activate it:
```
conda activate venv_quiche
```

The conda file includes the required R runtime and bridge packages (`r-base`, `rpy2`, `r-matrix`, `r-locfit`, `r-statmod`).

If you need to install `edgeR` in that activated conda environment, open R:
```
R
```

Then install:

```R
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

#statmod v1.5.0
install.packages('statmod')

#edger v3.40.2
BiocManager::install("edgeR")
```

### Quick smoke test
```bash
pytest tests/test_quiche_smoke.py -q
```

## Known limitations
- Differential enrichment paths that use edgeR/pertpy require optional heavy dependencies (`.[full]`) and, for edgeR-based workflows, a working R setup.
- `pip install .` intentionally does not install R/Rpy2 dependencies; use the conda environment when R-backed analysis is required.

## Example usage
In the subsections below, we will walk through a simple example of how you can use QUICHE. For a more detailed tutorial, including the necessary plotting functions, please see the demo Jupyter notebook [here](https://github.com/jranek/quiche/blob/main/notebooks/demo.ipynb). 

To perform spatial enrichment analysis with QUICHE, first load in the necessary packages.

```python
import os
import anndata
import quiche as qu
```

You can download an example dataset from the Zenodo repository by, 

```python
qu.pp.download_data(id = 'spain_preprocessed', overwrite = True)
```

Next, read in a preprocessed single-cell `.h5ad` object. The example `.h5ad` object contains multiple patient samples with TNBC profiled with MIBI-TOF imaging. Here, we're interested in identifying local cellular niches differentially-enriched in patients that do or do not relapse. 

```python
## load in data
adata = anndata.read_h5ad(os.path.join('data', 'spain_preprocessed.h5ad'))
adata.obs['Relapse'] = adata.obs['Relapse'].astype('int').astype('str')

## normalize expression data according to the modality of interest if this has not already been done. In this case, we'll just standardize the MIBI-TOF data by, 
adata.raw = adata
adata.X = qu.pp.standardize(adata.X)

## filter fovs with few cells
sketch_size = 1000
adata  = qu.pp.filter_fovs(adata, 'Patient_ID', sketch_size)
```

Then you can perform QUICHE spatial enrichment analysis by,

```python
## initialize class
quiche_op = qu.tl.QUICHE(adata = adata, labels_key = 'cell_cluster', spatial_key = 'spatial', fov_key = 'fov', patient_key = 'Patient_ID', segmentation_label_key = 'label')
## step 1: compute spatial niches 
quiche_op.compute_spatial_niches(radius = 200, n_neighbors = 30, min_cell_threshold = 3)
## step 2: perform distribution-focused downsampling
quiche_op.subsample(sketch_size = sketch_size, sketch_key = 'Patient_ID', n_jobs = 8)
## step 3: test for differential spatial enrichment across relapse conditions
quiche_op.differential_enrichment(design = '~Relapse', model_contrasts = 'Relapse1-Relapse0', k_sim = 100)
## step 4: annotate niche neighborhoods
quiche_op.annotate_niches(nlargest = 3, annotation_scheme = 'neighborhood', annotation_key = 'quiche_niche_neighborhood')
```

## License
This software is licensed under the MIT license (https://opensource.org/licenses/MIT).
