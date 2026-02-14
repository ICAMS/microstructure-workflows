[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ICAMS/microstructure-workflows.git/main?urlpath=%2Fdoc%2Ftree%2Febsd2kanapy2damask.ipynb)
# kanapy2DAMASK — Jupyter Workflow Setup Guide

This repository contains a complete workflow that connects **Kanapy** (microstructure generation) and **DAMASK** (FFT-based simulation). 

This notebook is accessible via [mybinder.org](https://mybinder.org/v2/gh/ICAMS/microstructure-workflows.git/main?urlpath=%2Fdoc%2Ftree%2F)  


To create a local copy on your hardware, follow the steps below.

---

## 1. Clone the repository

```bash
git clone https://gitlab.ruhr-uni-bochum.de/Workflows/kanapy2damask.git
```

## 2. Create the Conda environment

Change to the repository folder and create the environment using the provided file:

```bash
conda env create -f environment.yml
```

> This environment installs:  
> - DAMASK (from **conda-forge**)  
> - The **latest Kanapy** directly from GitHub  
> - The **damask\_python module** from the GitHub/ICAMS fork with patches for interfacing with Kanapy  
> - pyiron_workflow, orix, and all visualization tools  


## 3. Activate the environment

```bash
$ conda activate ms-data
```


## 4. Launch Jupyter

```bash
(ms-data) $ jupyter lab
```

Open

```
ebsd2kanapy2damask.ipynb
```

## 5. Run the notebook

After completing the steps above, open the notebook in JupyterLab and execute the cells sequentially (1 → 8).

The notebook will:

1. Load the EBSD file.  
2. Extract microstructure statistics.  
3. Generate a voxelized RVE with Kanapy.  
4. Build the data schema and DAMASK input files.  
5. Run the DAMASK simulation.  
6. Post-process the results (plots + VTK export).


