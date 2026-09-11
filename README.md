[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ICAMS/microstructure-workflows.git/main?urlpath=%2Fdoc%2Ftree%2Findex.ipynb)

# Microstructure Workflows — Kanapy → MiMeDO → DAMASK

This repository has been developed within the Infrastructure Use Case ([IUC07](https://nfdi-matwerk.de/about/nfdi-matwerk-structure/use-cases/iuc07)) *Beyond 3D: Tools for tracking spatiotemporal microstructure evolution* of the [NFDI-MatWerk](https://nfdi-matwerk.de) consortium.

It contains reusable, standardised workflows that connect **Kanapy** (3D synthetic microstructure generation and analysis) and **DAMASK** (crystal-plasticity simulation with the spectral/FFT solver) through the **MiMeDO** data object, orchestrated with **pyiron_workflow**. Starting from an experimental EBSD map, the workflows generate a statistically equivalent RVE, simulate its deformation, and track how the microstructure evolves — grain by grain — during processing and loading.

Start with the project guide: **[`index.ipynb`](index.ipynb)** (also readable on [mybinder.org](https://mybinder.org/v2/gh/ICAMS/microstructure-workflows.git/main?urlpath=%2Fdoc%2Ftree%2Findex.ipynb) — note that Binder cannot run the DAMASK solver, so use it to read, not to execute).

---

## Repository layout

```
microstructure-workflows/
├── index.ipynb                     <- project guide: concepts, nodes, identifiers, cases
├── environment.yml                 <- conda environment (mimedat)
├── images/                         <- figures and the video used by the guide
├── functions/                      <- ALL shared code
│   ├── nodes.py                    <- the pyiron_workflow pipeline nodes
│   ├── macros.py                   <- reusable multi-node chunks (simulate_case, simulate_branch)
│   ├── mimedo_identifier.py        <- deterministic identifiers for MiMeDO objects & orientations
│   ├── mimedo_viewer.py            <- interactive / static viewer for a MiMeDO JSON
│   ├── segment_microstructure.py   <- misorientation-based grain segmentation
│   └── voxel_graph.py              <- voxel neighbour graph (used by the viewer)
└── cases/                          <- one folder per study, wired from the shared nodes
    ├── Cold Rolling/
    └── Tensile and Rolling/
```

Everything reusable lives in `functions/`; a case folder holds only its notebook, its EBSD map and (after a run) its generated `Keys/` results. A fix in a node therefore only has to happen once.

## The cases

| Case | What it does |
|---|---|
| [**Cold Rolling**](cases/Cold%20Rolling/Cold%20Rolling.ipynb) | The reference case. A 316L EBSD map becomes a synthetic 3D RVE with the EBSD-derived texture, is exported as a MiMeDO object, compressed 30 % in X (plane strain) with DAMASK, then re-segmented and grain-tracked to capture the microstructure evolution. One linear pipeline: `load_ebsd_map → get_stats → generate_rve → write_data → load_to_damask → run_damask → post_processing`. |
| [**Tensile and Rolling**](cases/Tensile%20and%20Rolling/Tensile%20and%20Rolling.ipynb) | Seven DAMASK runs from one EBSD map, in one `wf.run()`. Branches A–C: uniaxial tension in X, Y, Z on the as-generated RVE. Branch D: cold rolling in X with regridding. Branches E–G: the same three tensile tests on the *rolled* microstructure D produced. Comparing A/E, B/F, C/G shows what rolling did to the mechanical anisotropy. Built from the `simulate_case` / `simulate_branch` macros. |

Every run writes its inputs and results to `cases/<case>/Keys/<key>/`, where `<key>` is a deterministic identifier composed from the MiMeDO metadata, the orientation set and the microstructure state — so related runs are recognisable at a glance and identical inputs map to the same folder. `Keys/` is git-ignored.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ICAMS/microstructure-workflows.git
cd microstructure-workflows
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
```

This installs:
- **DAMASK** from conda-forge (provides the `DAMASK_grid` solver binary),
- the latest **Kanapy** directly from GitHub,
- the **damask_python** module from the [ICAMS fork](https://github.com/ICAMS/damask_python), which overrides the conda-forge Python module with the MiMeDO interface (`load_MiMedat`, `from_mechanical_bc`, `Result.export_mimedo`),
- **pyiron_workflow**, **orix**, and the visualisation stack (matplotlib, plotly, pyvista, ipywidgets).

### 3. Activate and launch

```bash
conda activate mimedat
jupyter lab
```

Open `index.ipynb` first, then a case notebook. Each case notebook makes the shared code importable with two lines at the top:

```python
sys.path.insert(0, str(Path.cwd().parents[1]))   # repository root
from functions.nodes import load_ebsd_map, get_stats, generate_rve  # ...
```

### 4. Run a case

Execute the cells top to bottom. The `run_damask` node launches the real `DAMASK_grid` solver, so that step takes real compute time (minutes for the default 15³-voxel RVE, longer for Tensile and Rolling's seven runs). Reduce `nvox` in the RVE settings for a quick wiring check.

## Adding a case

Copy `cases/Cold Rolling/` (without `Keys/`) to a new, descriptively named folder, change only the parameters and the wiring, and add a card for it in `index.ipynb`. Anything that turns out to be reusable belongs in `functions/` — not copied into the case folder.

## Background

A central challenge in microstructure modelling is that different simulation tools store and represent data in incompatible formats. This makes it hard to build end-to-end workflows where the output of one tool becomes the input of another, and harder still to *track* microstructure evolution, since most tools only record their own internal state and lack a common structure for time-dependent changes in grains, phases and voxel fields.

The **MiMeDO** (Microstructure–Mechanics Data Object) format used here is that common structure: Kanapy writes it, DAMASK reads it and appends the deformed state to it, and the post-processing nodes re-segment and grain-track the result inside the same object.

### Integrated tools

- [**Kanapy**](https://icams.github.io/Kanapy/builds/html/index.html) — generates synthetic 3D microstructures from EBSD statistics and performs statistical analysis
- [**DAMASK**](https://damask-multiphysics.org) — crystal-plasticity simulation with the spectral (FFT) solver
- [**pyiron_workflow**](https://pyiron.org) — constructs the workflows as computational graphs
- [**orix**](https://orix.readthedocs.io) — crystal orientations and EBSD handling
