# kanapy2DAMASK — Jupyter Workflow Setup Guide

This repository contains a complete workflow that connects **Kanapy** (microstructure generation) and **DAMASK** (FFT-based simulation).  
Follow the steps below to create the environment and run the notebook.

---

## 🧭 1. Clone the repository

```bash
git clone https://gitlab.ruhr-uni-bochum.de/Workflows/kanapy2damask.git
```

---

## 📂 2. Create the Conda environment

Change to the repository folder and create the environment using the provided file:

```bash
cd kanapy2damask/env
conda env create -f environment.yml
```

> ✅ This environment installs:
> - DAMASK (from **conda-forge**)  
> - The **latest Kanapy** directly from GitHub  
> - pyiron_workflow, orix, and all visualization tools  

---

## ⚙️ 3. Activate the environment

```bash
conda activate kanapy2damask
```

---

## 💻 4. Register the Jupyter kernel

Register the environment with JupyterLab:

```bash
python -m ipykernel install --sys-prefix --name kanapy2damask --display-name "Python (kanapy2damask)"
```

> You can also use `--user` instead of `--sys-prefix` if you prefer user-level installation.

---

## 🚀 5. Launch JupyterLab

```bash
jupyter lab
```

Then navigate to:

```
kanapy2damask/notebooks
```

This folder contains the main notebook:

```
ebsd2kanapy2damask.ipynb
```

---

## 🧱 6. Prepare the EBSD map

Copy the EBSD file so that it sits **next to the notebook**:

```bash
cp ../data/ebsd_316L_500x500.ang .
```

After copying, you should have both files in the same directory:

```
kanapy2damask/notebooks/
 ├── ebsd2kanapy2damask.ipynb
 └── ebsd_316L_500x500.ang
```

---

## 🧩 7. Replace modified DAMASK modules

In this repository under `kanapy2damask/scripts/`, you’ll find two modified files:

- `_configmaterial.py` → replaces `ConfigMaterial`
- `_geomgrid.py` → replaces `GeomGrid`

Copy them into your **DAMASK installation inside the conda environment**, overwriting the existing ones.

```bash
# From repo root:
cp scripts/_configmaterial.py  ~/miniforge3/envs/kanapy2damask/lib/python3.11/site-packages/damask/ConfigMaterial.py
cp scripts/_geomgrid.py        ~/miniforge3/envs/kanapy2damask/lib/python3.11/site-packages/damask/GeomGrid.py
```

> ⚠️ Note: adjust the path if your conda installation is not under `~/miniforge3/`.

---

## ▶️ 8. Run the notebook

After completing the steps above, open the notebook in JupyterLab and execute the cells sequentially (1 → 8).

The notebook will:

1. Load the EBSD file.  
2. Extract microstructure statistics.  
3. Generate a voxelized RVE with Kanapy.  
4. Build the data schema and DAMASK input files.  
5. Run the DAMASK simulation.  
6. Post-process the results (plots + VTK export).

---

### ✅ You’re ready to go!
