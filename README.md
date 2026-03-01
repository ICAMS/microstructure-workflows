[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ICAMS/microstructure-workflows.git/main?urlpath=%2Fdoc%2Ftree%2Febsd2kanapy2damask.ipynb)
# kanapy2DAMASK — Jupyter Workflow Setup Guide

This repository has been developed within the Infrastaructure Use Case ([IUC07](https://nfdi-matwerk.de/about/nfdi-matwerk-structure/use-cases/iuc07)) *Beyond 3D: Tools for tracking spatiotemporal microstructure evolution* within the [NFDI MatWerk](https://nfdi-matwerk.de) project.
It contains a complete workflow that connects **Kanapy** (3D microstructure generation and analysis) and **DAMASK** (crystal plasticity with spectral solver (CPFFT)-based simulation of large deformations) to study the microstructure evolution during a simulated cold-rolling step of a synthetic polycrystal derived from an experimental EBSD map, see also Section 6. Background of the README file. 

This notebook is accessible via [mybinder.org](https://mybinder.org/v2/gh/ICAMS/microstructure-workflows.git/main?urlpath=%2Fdoc%2Ftree%2Febsd2kanapy2damask_new.ipynb)  


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


## 6. Background
A central challenge in microstructure modeling is that different simulation tools store and represent data in incompatible formats. This fragmentation makes it difficult to build end-to-end workflows where the output of one tool becomes the input of another. The problem becomes even more pronounced when tracking microstructure evolution, since most tools only record their own internal state and lack a common structure for storing time-dependent changes in grains, phases, and voxel fields.

### Integrated tools used in this notebook

[**Kanapy:**](https://icams.github.io/Kanapy/builds/html/index.html) generates synthetic 3D microstructures and performs statistical analysis  
[**DAMASK:**](https://damask-multiphysics.org) performs mechanical simulations based on FFT solver with crystal-plasticity model  
[**Pyiron_workflow:**](https://pyiron.org) constructs workflows as computational graphs
This notebook demonstrates how to construct and execute a complete microstructure-to-simulation workflow. 