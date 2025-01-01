# QUICHE

QUantitative InterCellular nicHe Enrichment

## Introduction

QUICHE is a statistical differential abundance testing method that can be used to discover cellular niches differentially enriched in spatial regions, longitudinal samples, or clinical patient groups. For more details on the method, please read the associated preprint: [Ranek JS, Greenwald NF, Goldston M, Camacho Fullaway C, Sowers C, Kong A, Mouron S, Quintela-Fandino M, West RB, Angelo M. QUICHE reveals structural definitions of anti-tumor responses in triple negative breast cancer. 2024]().

<p>
  <img src="https://github.com/jranek/quiche/blob/main/pipeline.png?raw=True" />
</p>

This repo is currently under development as we are in the process of porting over our existing code into this independent repository. In the meantime, you can access the code associated with the paper [here](https://github.com/angelolab/publications/tree/main/2024-Ranek_etal_QUICHE). 

## Data access
You can download all of the preprocessed MIBI-TOF datasets (`.h5ad` files) from the [Zenodo](https://zenodo.org/records/14290163) repository. Imaging data and cell segmentation masks can be found in the [BioStudies](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1507) repository. 

## Installation
You can clone the git repository by, 
```
git clone https://github.com/jranek/quiche.git
```
Then change the working directory as, 
```
cd quiche
```

For installation, we recommend that you create a conda environment using the provided yml file.

```
conda env create -f venv_quiche.yml
```

Once the environment is created, you can activate it by,
```
conda activate venv_quiche
```

In order to perform spatial enrichment analysis with QUICHE, you'll also need to install the necessary R packages.

```R
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

#statmod v1.5.0
install.packages('statmod')

#edger v3.40.2
BiocManager::install("edgeR")
```

## Example usage

## License
This software is licensed under the MIT license (https://opensource.org/licenses/MIT).

