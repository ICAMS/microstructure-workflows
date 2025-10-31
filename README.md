# kanapy2DAMASK — Jupyter Workflow Setup Guide

This repository contains a complete workflow that connects **Kanapy** (microstructure generation) and **DAMASK** (FFT-based simulation).  
Follow the steps below to create the environment and run the notebook.

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
> - pyiron_workflow, orix, and all visualization tools  


## 3. Activate the environment

```bash
$ conda activate kanapy2damask
```


## 4. Install local DAMASK version


```bash
(kanapy2damask) $ cd damask_python
(kanapy2damask) $ python -m pip install .
(kanapy2damask) $ cd ..
```


## 5. Launch Jupyter

```bash
(kanapy2damask) $ jupyter lab
```

Open

```
ebsd2kanapy2damask.ipynb
```

## 8. Run the notebook

After completing the steps above, open the notebook in JupyterLab and execute the cells sequentially (1 → 8).

The notebook will:

1. Load the EBSD file.  
2. Extract microstructure statistics.  
3. Generate a voxelized RVE with Kanapy.  
4. Build the data schema and DAMASK input files.  
5. Run the DAMASK simulation.  
6. Post-process the results (plots + VTK export).


