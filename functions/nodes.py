"""
Shared pyiron_workflow nodes for microstructure-to-simulation workflows.

This module is the single source of truth for the node functions reused
across experiment notebooks under ``cases/`` (tension, compression, cyclic
loading, and future cases). It currently holds Kanapy-side nodes (EBSD/stats
to RVE to MiMeDO) and DAMASK-side nodes (MiMeDO to DAMASK input/run/post-
processing). It may be split into per-solver modules (e.g. ``kanapy_nodes.py``,
``damask_nodes.py``, ``abaqus_nodes.py``) once enough solvers are added to
justify it.

Notebooks should import nodes from here rather than redefining them, so a
fix or improvement only needs to happen once.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import platform
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import damask
from pyiron_workflow import Workflow
import kanapy as knpy

from .mimedo_identifier import create_mimedo_identifier, create_orientation_identifier
from .segment_microstructure import segment_microstructure

# =========================
# Console formatting (ANSI)
# - Use OLD for bold, RESET to clear styles
# - Colors: CYAN / YELLOW / RED / GREEN
# =========================
OLD = "\033[1m"
CYAN = "\033[1;36m"
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
GREEN = "\033[1;32m"
RESET = "\033[0m"


###########################################################################
################### Load and read EBSD map using Kanapy ###################
###########################################################################
@Workflow.wrap.as_function_node("ebsd_map")
def load_ebsd_map(
    file_path: Union[str, Path],
    gs_min: float = 10.0,
    vf_min: float = 0.03,
    max_angle: float = 5.0,
    connectivity: int = 8,
    show_plot: bool = False,
    show_hist: Optional[bool] = None,
    show_grains: bool = False,
    felzenszwalb: bool = False,
) -> Any:
    """
    Load an EBSD map and segment it into grains using Kanapy.

    Thin wrapper around ``kanapy.EBSDmap``. All segmentation-affecting
    parameters (`gs_min`, `vf_min`, `max_angle`, `connectivity`) are passed
    through directly to Kanapy; only the deprecated `hist`/`plot` aliases are
    dropped in favor of their non-deprecated equivalents.

    Parameters
    ----------
    file_path : str or Path
        Relative or absolute path to the EBSD file (e.g. ``.ang``, ``.ctf``).
    gs_min : float, optional
        Minimum grain size in pixels. Grains smaller than this are merged
        into their largest neighboring grain during segmentation.
        Default is 10.0.
    vf_min : float, optional
        Minimum phase volume fraction required for a phase to be kept.
        Phases below this threshold are discarded. Default is 0.03.
    max_angle : float, optional
        Misorientation angle tolerance in degrees used when merging
        neighboring pixels/regions into the same grain. Default is 5.0.
    connectivity : int, optional
        Pixel neighbor connectivity used by the region-growing segmentation
        algorithm. Default is 8.
    show_plot : bool, optional
        If True, display the misorientation map, IPF map, segmentation map,
        and pole figure. Default is False.
    show_hist : bool or None, optional
        If True, display the grain-equivalent-diameter histogram. If None,
        follows `show_plot`. Default is None.
    show_grains : bool, optional
        If True, display an additional grain map plot. Default is False.
    felzenszwalb : bool, optional
        If True, display a Felzenszwalb-algorithm comparison plot. This is a
        diagnostic visualization only; it does not change segmentation
        results. Default is False.

    Returns
    -------
    kanapy.EBSDmap
        The loaded and segmented EBSD map object.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not resolve to an existing file.
    """
    print(f"{CYAN}{OLD}########################################################################################{RESET}")
    print(f"{YELLOW}{OLD} Loading EBSD map ....................................................................{RESET}")
    print("\n\n")

    cwd = os.getcwd()
    full_path = os.path.join(cwd, file_path)

    if not os.path.exists(full_path):
        print(f"{RED}{OLD}ERROR: EBSD file not found in current directory.{RESET}")
        print(f"{YELLOW}{OLD}  -> Current working directory: {cwd}{RESET}")
        print(f"{YELLOW}{OLD}  -> Expected file: {file_path}{RESET}")
        raise FileNotFoundError(f"File '{file_path}' does not exist in the current directory.")

    ebsd = knpy.EBSDmap(
        full_path,
        gs_min=gs_min,
        vf_min=vf_min,
        max_angle=max_angle,
        connectivity=connectivity,
        show_plot=show_plot,
        show_hist=show_hist,
        show_grains=show_grains,
        felzenszwalb=felzenszwalb,
    )

    print(f"{GREEN}{OLD}EBSD file loaded successfully!{RESET}")
    print(f"{YELLOW}{OLD}  -> Current working directory: {cwd}{RESET}")
    print(f"{YELLOW}{OLD}  -> EBSD file: {file_path}{RESET}")
    print("\n\n")
    print(f"{YELLOW}{OLD}EBSD map loading is completed ........................................................{RESET}")
    print(f"{CYAN}{OLD}########################################################################################{RESET}")
    print("\n\n")

    return ebsd


#####################################################################################################
#################### Extract statistical parameters from EBSD map or a JSON file ####################
#####################################################################################################
@Workflow.wrap.as_function_node("statisticalDescriptors")
def get_stats(
    source: Any,
    NVoxels: int = 25,
    sizeRVE: int = 25,
    periodic: bool = True,
    deq_min: Optional[float] = None,
    deq_max: Optional[float] = None,
    asp_min: Optional[float] = None,
    asp_max: Optional[float] = None,
    omega_min: Optional[float] = None,
    omega_max: Optional[float] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract statistical microstructure descriptors from an EBSD map or a JSON file.

    Parameters
    ----------
    source : Any
        Either a Kanapy EBSD map object returned by ``load_ebsd_map`` or a path to
        a JSON file of precomputed statistics. The type hint is intentionally broad
        because pyiron_workflow validates node connections using function
        annotations.
    NVoxels : int, optional
        Number of voxels per axis used for discretization. Default is 25.
    sizeRVE : int, optional
        Target physical size of the representative volume element (RVE).
        Default is 25.
    periodic : bool, optional
        Whether the generated RVE should be periodic. Default is True.
    deq_min, deq_max : float or None, optional
        Cutoff bounds for the equivalent-diameter distribution, passed through
        to ``knpy.set_stats``. If None, Kanapy computes its own data-driven
        default from the phase's lognormal fit. Default is None.
    asp_min, asp_max : float or None, optional
        Cutoff bounds for the aspect-ratio distribution, passed through to
        ``knpy.set_stats``. If None, Kanapy computes its own default.
        Default is None.
    omega_min, omega_max : float or None, optional
        Cutoff bounds for the tilt-angle distribution, passed through to
        ``knpy.set_stats``. If None, Kanapy computes its own default
        (``[-pi, pi]``). Default is None.

    Returns
    -------
    dict or list of dict
        - For JSON input: whatever structure is returned by ``knpy.import_stats``.
        - For EBSD input with one phase: a single statistics dictionary.
        - For EBSD input with multiple phases: a list of statistics dictionaries,
          one per phase, suitable for use as
          ``knpy.Microstructure(descriptor=[ms_stats_0, ms_stats_1, ...])``.

    Notes
    -----
    All cutoff parameters default to ``None`` so that Kanapy's own
    data-driven defaults apply unless explicitly overridden. Earlier
    versions of this node hardcoded fixed cutoff values tuned for one
    specific EBSD sample; that behavior is intentionally not reproduced
    here so the node generalizes to any EBSD input.
    """
    print(f"{CYAN}{OLD}########################################################################################{RESET}")
    print(f"{YELLOW}{OLD}Extracting microstructure statistics .................................................{RESET}")
    print("\n\n")

    # -------------------------------------------------------------------------
    # Case 1: If the source is a JSON file, import precomputed statistics
    # -------------------------------------------------------------------------
    if isinstance(source, (str, Path)):
        src = str(source)
        if not src.lower().endswith(".json"):
            raise ValueError("Only JSON files can be imported as statistical parameters.")
        if not os.path.isfile(src):
            raise FileNotFoundError(f"The specified JSON file '{src}' does not exist.")
        try:
            ms_stats = knpy.import_stats(src)
        except Exception as error:
            raise ValueError(f"Error reading JSON file '{src}': {error}")

    # -------------------------------------------------------------------------
    # Case 2: If the source is Kanapy EBSDmap-like object, compute statistics
    # -------------------------------------------------------------------------
    else:
        # Validation for expected attributes
        if not hasattr(source, "ms_data"):
            raise ValueError("Invalid EBSD map object: missing attribute 'ms_data'.")

        ms_data_list = source.ms_data
        if not ms_data_list:
            raise ValueError("EBSD map object contains empty 'ms_data'.")

        phase_descriptors = []

        try:
            for ph_index, ph_data in enumerate(ms_data_list):
                # Normalize phase name: lowercase + underscores
                matname = ph_data["name"].lower().replace(" ", "_")

                gs_param = ph_data["gs_param"]  # log-normal parameters for grain size
                ar_param = ph_data["ar_param"]  # log-normal parameters for aspect ratios
                om_param = ph_data["om_param"]  # normal distribution for tilt angles

                # Compute statistics for this phase using Kanapy
                phase_stats = knpy.set_stats(
                    gs_param,
                    ar_param,
                    om_param,
                    deq_min=deq_min, deq_max=deq_max,
                    asp_min=asp_min, asp_max=asp_max,
                    omega_min=omega_min, omega_max=omega_max,
                    voxels=NVoxels,
                    size=sizeRVE,
                    periodicity=periodic,
                    VF=ph_data["vf"],
                    phasename=matname,
                    phasenum=ph_index,
                )

                # Attach raw data for traceability (optional but useful)
                phase_stats["Data"] = {
                    "grain_size": ph_data.get("gs_data"),
                    "aspect_ratio": ph_data.get("ar_data"),
                }

                phase_descriptors.append(phase_stats)

        except (KeyError, IndexError, AttributeError) as exc:
            raise ValueError(f"EBSD map object is missing expected fields: {exc}")

        # For single-phase case, keep backward-compatible behavior
        if len(phase_descriptors) == 1:
            ms_stats = phase_descriptors[0]
        else:
            ms_stats = phase_descriptors

    print("\n\n")
    print(f"{YELLOW}{OLD}Microstructure statistics extracting is completed .................................{RESET}")
    print(f"{CYAN}{OLD}#####################################################################################{RESET}")
    print("\n\n")

    return ms_stats


########################################################################################################
#################### Create a state for simulating the microstructure of materials ####################
########################################################################################################
@Workflow.wrap.as_function_node("statisticalDescriptors")
def create_stats(
    case_type: str,
    phases: List[Dict[str, Any]],
    length_side: float,
    number_voxels: int,
    periodicity: bool,
    unit: str = "um",
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Create Kanapy-compatible microstructure descriptor(s) from user-specified
    (not EBSD-derived) statistical parameters.

    This node supports three cases:

    1. single_phase
       Dense one-phase polycrystal.
       Output: one descriptor dictionary.
       Requirement: exactly one phase with volume_fraction = 1.0.

    2. multi_phase
       Dense multiphase polycrystal.
       Output: list of descriptor dictionaries.
       Requirement: two or more phases and sum(volume_fraction) = 1.0.

    3. inclusion
       Sparse inclusion / pore / precipitate phase inside an automatic matrix.
       Output: one descriptor dictionary.
       Requirement: exactly one sparse phase with 0 < volume_fraction < 1.

    Parameters
    ----------
    case_type : str
        One of:
        - "single_phase"
        - "multi_phase"
        - "inclusion"

    phases : list of dict
        Phase definitions controlled by the user.

        Required keys per phase:
        - "name" : str
            Phase name, e.g. "Ferrite", "Martensite", "Inclusions".
        - "volume_fraction" : float
            Phase volume fraction.
        - "grain_type" : str
            Either "Equiaxed" or "Elongated".
        - "grain_size" : float
            Equivalent diameter scale.

        Optional keys per phase:
        - "number" : int
            Phase number. If not given, the phase index is used.

        Optional equivalent-diameter controls (shape parameters, always used):
        - "equiv_sig" : float, default 0.7
        - "equiv_loc" : float, default 0.0

        Optional equivalent-diameter cutoffs (passed through to
        ``knpy.set_stats``; if omitted, Kanapy computes its own data-driven
        default from `equiv_sig`/`equiv_loc`/`grain_size` instead of a fixed
        formula):
        - "equiv_cutoff_min" : float
        - "equiv_cutoff_max" : float

        Optional elongated-grain shape controls (always used when
        `grain_type` is "Elongated"):
        - "aspect_sig" : float, default 0.8
        - "aspect_loc" : float, default 0.0
        - "aspect_scale" : float, default 5.0
        - "tilt_kappa" : float, default 0.5
        - "tilt_loc" : float, default pi / 2

        Optional elongated-grain cutoffs (passed through to
        ``knpy.set_stats``; if omitted, Kanapy computes its own default):
        - "aspect_cutoff_min" : float
        - "aspect_cutoff_max" : float
        - "tilt_cutoff_min" : float
        - "tilt_cutoff_max" : float

    length_side : float
        Side length of the cubic RVE.

    number_voxels : int
        Number of voxels along each RVE side.

    periodicity : bool
        Whether the RVE should be periodic.

    unit : str, optional
        Kanapy output unit. Use "um" or "mm".
        "μm" and "µm" are converted to "um".

    Returns
    -------
    source : dict or list of dict
        Kanapy descriptor source for ``knpy.Microstructure``.

        - single_phase -> dict
        - inclusion    -> dict
        - multi_phase  -> list[dict]

    Notes
    -----
    Per-phase dict construction is delegated to ``knpy.set_stats`` rather
    than reimplemented here, so schema and default-cutoff behavior always
    track Kanapy directly. Only cutoffs a phase dict explicitly supplies are
    passed through; anything omitted is left as ``None`` so Kanapy's own
    default applies -- the same policy used by ``get_stats``, so both
    Kanapy-input paths behave consistently for equivalent inputs.

    This function is written as a pyiron_workflow-safe node:
    - no nested helper functions,
    - no early return statements,
    - one final return statement only.
    """

    print(f"{CYAN}{OLD}########################################################################################{RESET}")
    print(f"{YELLOW}{OLD}Step 1: Generate Kanapy microstructure descriptor(s) .................................{RESET}")

    # -------------------------------------------------------------------------
    # 1) Normalize and validate global inputs
    # -------------------------------------------------------------------------
    case_type = str(case_type).strip().lower()

    if case_type not in {"single_phase", "multi_phase", "inclusion"}:
        raise ValueError(
            "`case_type` must be one of: "
            "'single_phase', 'multi_phase', or 'inclusion'. "
            f"Got {case_type!r}."
        )

    unit = str(unit).strip().replace("μ", "u").replace("µ", "u")

    if unit not in {"um", "mm"}:
        raise ValueError(
            f"Kanapy output unit must be 'um' or 'mm'. Got {unit!r}."
        )

    if not isinstance(phases, list) or len(phases) == 0:
        raise ValueError("`phases` must be a non-empty list of dictionaries.")

    if length_side <= 0:
        raise ValueError(f"`length_side` must be positive. Got {length_side}.")

    if number_voxels <= 0:
        raise ValueError(
            f"`number_voxels` must be positive. Got {number_voxels}."
        )

    # -------------------------------------------------------------------------
    # 2) Validate phase definitions
    # -------------------------------------------------------------------------
    volume_fraction_sum = 0.0
    phase_numbers = []

    for i, phase in enumerate(phases):
        if not isinstance(phase, dict):
            raise ValueError(f"Phase entry {i} must be a dictionary.")

        required_keys = ["name", "volume_fraction", "grain_type", "grain_size"]

        for key in required_keys:
            if key not in phase:
                raise ValueError(f"Phase entry {i} is missing required key {key!r}.")

        phase_name = str(phase["name"])
        grain_type = str(phase["grain_type"]).strip().lower()
        grain_size = float(phase["grain_size"])
        volume_fraction = float(phase["volume_fraction"])
        phase_number = int(phase.get("number", i))

        if grain_type not in {"equiaxed", "elongated"}:
            raise ValueError(
                f"Phase {phase_name!r}: `grain_type` must be "
                "'Equiaxed' or 'Elongated'. "
                f"Got {phase['grain_type']!r}."
            )

        if grain_size <= 0:
            raise ValueError(
                f"Phase {phase_name!r}: `grain_size` must be positive. "
                f"Got {grain_size}."
            )

        if not (0.0 < volume_fraction <= 1.0):
            raise ValueError(
                f"Phase {phase_name!r}: `volume_fraction` must be in (0, 1]. "
                f"Got {volume_fraction}."
            )

        if phase_number in phase_numbers:
            raise ValueError(
                f"Duplicate phase number {phase_number}. "
                "Each phase must have a unique phase number."
            )

        phase_numbers.append(phase_number)
        volume_fraction_sum += volume_fraction

    # -------------------------------------------------------------------------
    # 3) Validate case-specific logic
    # -------------------------------------------------------------------------
    if case_type == "single_phase":
        if len(phases) != 1:
            raise ValueError(
                "`single_phase` requires exactly one phase definition."
            )

        if not np.isclose(volume_fraction_sum, 1.0):
            raise ValueError(
                "`single_phase` requires volume_fraction = 1.0. "
                f"Got {volume_fraction_sum}."
            )

    if case_type == "multi_phase":
        if len(phases) < 2:
            raise ValueError(
                "`multi_phase` requires at least two phase definitions."
            )

        if not np.isclose(volume_fraction_sum, 1.0):
            raise ValueError(
                "`multi_phase` requires the sum of phase volume fractions "
                f"to be 1.0. Got {volume_fraction_sum}."
            )

        if len(phases) > 2:
            print(
                f"{YELLOW}Warning: More than two phases were provided. "
                f"Kanapy multiphase examples are mainly two-phase cases. "
                f"Check the generated RVE carefully.{RESET}"
            )

    if case_type == "inclusion":
        if len(phases) != 1:
            raise ValueError(
                "`inclusion` mode expects exactly one sparse phase definition. "
                "Kanapy creates the matrix phase automatically."
            )

        if not (0.0 < volume_fraction_sum < 1.0):
            raise ValueError(
                "`inclusion` mode requires 0 < volume_fraction < 1. "
                f"Got {volume_fraction_sum}."
            )

    # -------------------------------------------------------------------------
    # 4) Build Kanapy descriptors by delegating to knpy.set_stats
    # -------------------------------------------------------------------------
    descriptors = []

    for i, phase in enumerate(phases):
        phase_name = str(phase["name"])
        phase_number = int(phase.get("number", i))
        volume_fraction = float(phase["volume_fraction"])

        grain_type_raw = str(phase["grain_type"]).strip().lower()
        grain_type = "Equiaxed" if grain_type_raw == "equiaxed" else "Elongated"

        grain_size = float(phase["grain_size"])

        grains = [
            float(phase.get("equiv_sig", 0.7)),
            float(phase.get("equiv_loc", 0.0)),
            grain_size,
        ]

        ar = None
        omega = None
        if grain_type == "Elongated":
            ar = [
                float(phase.get("aspect_sig", 0.8)),
                float(phase.get("aspect_loc", 0.0)),
                float(phase.get("aspect_scale", 5.0)),
            ]
            omega = [
                float(phase.get("tilt_kappa", 0.5)),
                float(phase.get("tilt_loc", 0.5 * np.pi)),
            ]

        descriptor = knpy.set_stats(
            grains=grains,
            ar=ar,
            omega=omega,
            deq_min=phase.get("equiv_cutoff_min"),
            deq_max=phase.get("equiv_cutoff_max"),
            asp_min=phase.get("aspect_cutoff_min"),
            asp_max=phase.get("aspect_cutoff_max"),
            omega_min=phase.get("tilt_cutoff_min"),
            omega_max=phase.get("tilt_cutoff_max"),
            size=length_side,
            voxels=number_voxels,
            gtype=grain_type,
            rveunit=unit,
            periodicity=periodicity,
            VF=volume_fraction,
            phasename=phase_name,
            phasenum=phase_number,
        )

        descriptors.append(descriptor)

    # -------------------------------------------------------------------------
    # 5) Select output structure expected by Kanapy
    # -------------------------------------------------------------------------
    if case_type == "multi_phase":
        source = descriptors
    else:
        source = descriptors[0]

    # -------------------------------------------------------------------------
    # 6) Print state summary
    # -------------------------------------------------------------------------
    print("\nDescriptor state:")
    print(f"  Case type: {case_type}")
    print(f"  Output type: {'list[dict]' if isinstance(source, list) else 'dict'}")
    print(f"  Number of descriptors: {len(descriptors)}")
    print(f"  Volume-fraction sum: {volume_fraction_sum}")
    print(f"  RVE side length: {length_side} {unit}")
    print(f"  Voxels per side: {number_voxels}")
    print(f"  Periodicity: {periodicity}")

    print("\nPhases:")
    for descriptor in descriptors:
        phase_block = descriptor["Phase"]
        equiv_block = descriptor["Equivalent diameter"]

        print(
            f"  - {phase_block['Name']}: "
            f"phase {phase_block['Number']}, "
            f"VF = {phase_block['Volume fraction']}, "
            f"{descriptor['Grain type']}, "
            f"grain size scale = {equiv_block['scale']} {unit}"
        )

    if case_type == "inclusion":
        print(
            "\nInclusion state:"
            "\n  Kanapy will treat this as a sparse phase."
            "\n  The remaining volume is handled as an automatic matrix phase."
        )

    print("\nKanapy descriptor source:")
    print(source)

    print("\n")
    print(f"{YELLOW}{OLD}Step 1 complete: descriptor source is created.......................................{RESET}")
    print(f"{CYAN}{OLD}######################################################################################{RESET}")
    print("\n\n")

    return source


######################################################################################################
#################### create an RVE for simulating the microstructure of materials ####################
######################################################################################################
@Workflow.wrap.as_function_node("RVE")
def generate_rve(
    source: Union[Dict[str, Any], List[Dict[str, Any]]],
    ori: Any,
    show_plots: bool = True,
    report_grain_geometry: bool = False,
    orientation_Nbase: int = 1000,
    verbose: bool = False,
) -> knpy.Microstructure:
    """
    Construct a 3D representative volume element (RVE) from microstructure
    statistical descriptors and assign crystallographic orientations.

    Parameters
    ----------
    source : dict or list of dict
        Microstructure descriptor produced by ``get_stats``/``create_stats``
        (or an equivalent ``knpy.set_stats`` output). Must contain phase
        information, geometric statistics, and voxelization settings under
        keys such as ``"Phase"``, ``"Equivalent diameter"``, ``"Aspect
        ratio"``, and ``"RVE"``.
    ori : str or object
        Orientation specification for the generated RVE. Accepted values:
        - "goss"     -> unimodal Goss texture
        - "copper"   -> unimodal Copper texture
        - "random"   -> fully random texture
        - a ``kanapy.EBSDmap`` object, to reuse a real EBSD-derived
          orientation distribution on this synthetic RVE geometry. This is
          the only other accepted type; a raw Euler-angle array is not.
    show_plots : bool, optional
        If True, display ellipsoid packing, voxelization slices, and
        orientation-colored voxel maps. Default is True.
    report_grain_geometry : bool, optional
        If True, compute polyhedral grain geometry (``ms.generate_grains()``)
        after voxelization, for statistical comparison between the target
        descriptor and the actually-achieved microstructure. This is not
        required by downstream MiMeDO export -- it is purely an optional
        validation step. When True and `show_plots` is also True, the
        initial-statistics plot, grain plot, and achieved-vs-target
        statistics plot are all shown. Default is False.
    orientation_Nbase : int, optional
        Number of base orientations used when generating textures (passed
        to ``ms.generate_orientations``). Default is 1000.
    verbose : bool, optional
        If True, print additional information during orientation
        generation. Default is False.

    Returns
    -------
    knpy.Microstructure
        A fully generated microstructure object containing:
        - packed ellipsoidal grains,
        - voxelized 3D grid representation,
        - assigned crystallographic orientations,
        - plotting and export utilities for downstream simulations.

    Notes
    -----
    For the custom-orientation branch (`ori` not a string), this node calls
    ``ms.generate_orientations(ori, Nbase=orientation_Nbase, verbose=verbose)``.
    Earlier versions of this node called that branch with `res_low`,
    `res_high`, `lim`, and `hw_init` keyword arguments that do not exist on
    ``generate_orientations`` (those belong to an internal helper it calls,
    except `hw_init` which does not correspond to any real parameter); since
    every known caller so far only ever passed `ori="random"`, that branch
    was never actually exercised and the resulting ``TypeError`` never
    surfaced. It is fixed here.
    """
    print(f"{CYAN}{OLD}########################################################################################{RESET}")
    print(f"{YELLOW}{OLD}Generating RVE .......................................................................{RESET}")
    print("\n\n")

    if isinstance(source, dict):
        descriptor = source
        phase_block = source.get("Phase") or {}
        # Use phase name if present, otherwise a neutral default
        material_name = phase_block.get("Name") or "RVE"
        data_block = source.get("Data", {})
    elif isinstance(source, list):
        descriptor = source

        phase_names = []
        for ph in source:
            if isinstance(ph, dict):
                ph_block = ph.get("Phase") or {}
                ph_name = ph_block.get("Name")
                if ph_name:
                    phase_names.append(ph_name)
        if phase_names:
            material_name = "_".join(phase_names)
        else:
            material_name = "multi_phase_RVE"
        data_block = source[0].get("Data", {}) if source and isinstance(source[0], dict) else {}
    else:
        raise ValueError(
            "`source` must be a dict or a list of dicts "
            "(e.g., output of get_stats / create_stats / knpy.set_stats)."
        )

    is_multiphase = isinstance(source, list)
    gs_data = data_block.get("grain_size")
    ar_data = data_block.get("aspect_ratio")

    # -------------------------------------------------------------------------
    # 1) Create simulation box with ellipsoidal grains
    # -------------------------------------------------------------------------
    ms = knpy.Microstructure(descriptor=source, name=f"{material_name}_RVE")
    ms.init_RVE()
    # Creates particle distribution (within the given lower and upper bounds)
    # inside simulation box (RVE) based on the data provided in the data file
    # -------------------------------------------------------------------------

    # Optional: plot initial statistics (target distribution, optionally
    # compared against raw EBSD-derived data if present in `source["Data"]`).
    if report_grain_geometry and show_plots:
        if gs_data is not None and ar_data is not None:
            ms.plot_stats_init(gs_data=gs_data, ar_data=ar_data)
        else:
            ms.plot_stats_init()

    # -------------------------------------------------------------------------
    # 2) Perform particle packing
    # -------------------------------------------------------------------------
    ms.pack()
    print("\n\n")
    if show_plots:
        ms.plot_ellipsoids()
    # Simulates grain packing in the box. Particles are initially downscaled
    # to avoid overlap and allow free movement. Collisions are handled during packing.
    # ---------------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # 3) Voxelize to 3D grid
    # -------------------------------------------------------------------------
    ms.voxelize()
    print("\n\n")
    if show_plots:
        ms.plot_voxels(sliced=True)
    # Converts the packed structure into a voxelated 3D mesh.
    #   Voxels inside grains are assigned to the grain phase, while others
    #   are filled according to a grain growth algorithm.
    # --------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # 3b) Optional: compute polyhedral grain geometry and achieved statistics
    # -------------------------------------------------------------------------
    if report_grain_geometry:
        ms.generate_grains()
        if show_plots:
            ms.plot_grains(phases=is_multiphase)
            ms.plot_stats(show_all=True, phases=is_multiphase)

    # -------------------------------------------------------------------------
    # 4) Assign orientations to grains
    # -------------------------------------------------------------------------
    if isinstance(ori, str):
        key = ori.lower().strip()
        if key == "goss":
            ang, omega, texture_desc = [0, 45, 0], 7.5, "unimodal"
        elif key == "copper":
            ang, omega, texture_desc = [90, 35, 45], 7.5, "unimodal"
        elif key == "random":
            ang, omega, texture_desc = None, None, "random"
        else:
            raise ValueError('Unsupported texture string in `ori`. Use "goss", "copper", or "random".')

        ms.generate_orientations(
            texture_desc, ang=ang, omega=omega, Nbase=orientation_Nbase, verbose=verbose
        )
    else:
        # Custom orientation source: a kanapy.EBSDmap object, to reuse a
        # real EBSD-derived orientation distribution on this synthetic RVE.
        ms.generate_orientations(ori, Nbase=orientation_Nbase, verbose=verbose)
    print("\n\n")

    if show_plots:
        ms.plot_voxels(ori=True)  # plot voxelized grains in color code of IPF key

    print("\n\n")
    print(f"{YELLOW}{OLD}RVE generating is completed ........................................................{RESET}")
    print(f"{CYAN}{OLD}######################################################################################{RESET}")
    print("\n\n")

    return ms  # Return the generated microstructure object


##################################################################################################################
#################### write the data json file containing all the data required the simulation ####################
##################################################################################################################
@Workflow.wrap.as_function_node(
    "MiMeDO",
    "pathtoMiMeDO",
    "identifier",
    "key",
    "input_path",
    "results_path",
)
def write_data(
    source: knpy.Microstructure,
    user_metadata: dict,
    boundary_condition: dict,
    phases: list,
    units: dict,
    base_work_dir: Union[str, Path] = "Keys",
) -> Tuple[Dict[str, Any], str, str, str, str, str]:
    """
    Build and save a JSON data object (MiMeDO) from a Kanapy microstructure,
    and create its Keys/<key>/{inputs,results} workflow folder.

    Delegates core schema construction to ``source.write_data`` (Kanapy),
    enriches the result with system/software metadata, installs phase-wise
    Euler angle arrays and their orientation identifiers, computes the
    deterministic MiMeDO identifier, builds the folder key, and writes the
    data object to ``base_work_dir/<key>/results/<identifier>.json``.

    Parameters
    ----------
    source : kanapy.Microstructure
        RVE produced by ``generate_rve``. Orientations must already be
        assigned (``generate_rve`` always does this).
    user_metadata : dict
        Schema metadata fields (title, creator, dates, etc.).
    boundary_condition : dict
        Mechanical or thermal boundary-condition definition.
    phases : list of dict
        Phase definitions passed through to ``source.write_data``.
    units : dict
        Must contain a length entry, e.g. ``{"length": "um"}``.
    base_work_dir : str or Path, optional
        Base directory under which ``<key>/{inputs,results}`` is created.
        Default is ``"Keys"``.

    Returns
    -------
    tuple
        Workflow node outputs:

        - MiMeDO : dict
            Full MiMeDO data object.
        - pathtoMiMeDO : str
            Path to the saved JSON file.
        - identifier : str
            Deterministic MiMeDO identifier (data-object level).
        - key : str
            Folder key: ``identifier`` + one ``orientation_identifier`` per
            phase (in phase order) + the initial snapshot's
            ``microstructure_state_id`` -- joined with ``"_"``. Folders that
            share a segment share that piece of the run (same microstructure,
            same texture, or the same overall run), visible at a glance
            without opening any JSON.
        - input_path : str
            Folder for DAMASK input files (``inputs/``).
        - results_path : str
            Folder for DAMASK output files and the saved MiMeDO
            (``results/``).

    Notes
    -----
    ``identifier`` alone remains the single comprehensive fingerprint of the
    whole run (stored as ``data["identifier"]``) and is fixed once computed:
    it is reproducible even after later workflow stages (running the
    simulation, post-processing) append further snapshots/results, because
    it is bound to ``microstructure[0]`` rather than the whole, growing
    ``microstructure`` array. ``key`` is a separate, purpose-built label
    used only for the folder name, deliberately decomposed for legibility
    rather than being a single opaque hash.
    """

    print(f"{CYAN}{OLD}########################################################################################{RESET}")
    print(f"{YELLOW}{OLD}Exporting MiMeDO .......................................................................{RESET}")
    print("\n\n")

    # -------------------------------------------------------------------------
    # 1) Validate basic inputs
    # -------------------------------------------------------------------------
    if not isinstance(user_metadata, dict):
        raise ValueError("user_metadata must be a dict")

    if not isinstance(boundary_condition, dict):
        raise ValueError("boundary_condition must be a dict")

    if not isinstance(phases, (list, tuple)):
        raise ValueError("phases must be a list or tuple")

    if not isinstance(units, dict):
        raise ValueError("units must be a dict")

    if not isinstance(base_work_dir, (str, Path)):
        raise ValueError("base_work_dir must be a string or pathlib.Path")

    # -------------------------------------------------------------------------
    # 2) Normalize length unit
    # -------------------------------------------------------------------------
    units_norm = {str(k).lower(): v for k, v in units.items()}
    length_unit = units_norm.get("length")

    if length_unit is None:
        length_unit = units.get("Length") or units.get("LENGTH")

    if length_unit is None:
        raise ValueError(
            "units must contain a length entry, e.g. {'length': 'µm'}, "
            "{'length': 'um'}, {'length': 'mm'}, or {'length': 'm'}."
        )

    length_unit_str = str(length_unit).strip()

    if length_unit_str in {"µm", "μm", "um", "micrometer", "micrometre"}:
        length_unit = "µm"
    elif length_unit_str in {"mm", "millimeter", "millimetre"}:
        length_unit = "mm"
    elif length_unit_str in {"m", "meter", "metre"}:
        length_unit = "m"
    else:
        raise ValueError(
            "Unsupported length unit. Expected one of: 'µm', 'um', 'mm', or 'm'. "
            f"Got: {length_unit!r}"
        )

    # -------------------------------------------------------------------------
    # 3) Build data structure via Kanapy
    # -------------------------------------------------------------------------
    data = source.write_data(
        user_metadata=user_metadata,
        boundary_condition=boundary_condition,
        phases=list(phases),
        interactive=False,
        structured=True,
        length_unit=length_unit,
    )

    if not isinstance(data, dict) or len(data) == 0:
        raise RuntimeError("source.write_data did not return a non-empty dict.")

    # Add units to the data schema
    data["units"] = units

    # Keep provided date if available; otherwise add today's date.
    # Do not overwrite date every run, because date participates in the MiMeDO hash.
    if not data.get("date"):
        data["date"] = datetime.utcnow().strftime("%Y-%m-%d")

    # -------------------------------------------------------------------------
    # 4) Locate initial microstructure snapshot and validate grain orientations
    #
    # Kanapy's write_data always returns "microstructure" as a list of
    # time-step snapshots (microstructure[0] is the initial, simulation-ready
    # state); it also assigns microstructure[0]["microstructure_state_id"]
    # unconditionally before returning, so it is never absent here.
    # -------------------------------------------------------------------------
    microstructure_block = data.get("microstructure")

    if not isinstance(microstructure_block, list) or len(microstructure_block) == 0:
        raise ValueError(
            "Expected 'microstructure' as a non-empty list of snapshot dicts."
        )

    initial_microstructure = microstructure_block[0]

    if not isinstance(initial_microstructure, dict):
        raise ValueError("microstructure[0] must be a dict.")

    if not initial_microstructure.get("microstructure_state_id"):
        raise ValueError(
            "microstructure[0] is missing 'microstructure_state_id'. "
            "Expected Kanapy's write_data to assign it unconditionally."
        )

    grains = initial_microstructure.get("grains")

    if not isinstance(grains, list) or len(grains) == 0:
        raise ValueError("Missing or invalid 'grains' list inside 'microstructure[0]'.")

    for i, grain in enumerate(grains):
        if not isinstance(grain, dict):
            raise ValueError(
                f"Grain at index {i} is not a dict: {type(grain).__name__}"
            )

        orientation = grain.get("orientation")

        if not (isinstance(orientation, (list, tuple)) and len(orientation) == 3):
            raise ValueError(
                f"Grain {i} has invalid 'orientation'. "
                f"Expected [Phi1, Phi, Phi2], got: {orientation}"
            )

    # -------------------------------------------------------------------------
    # 5) Update software and system information
    # -------------------------------------------------------------------------
    data["software"] = "DAMASK"
    data["software_version"] = getattr(damask, "__version__", "unknown")

    data["system"] = platform.system()
    data["system_version"] = platform.version()
    data["processor_specifications"] = platform.processor() or platform.machine()

    # -------------------------------------------------------------------------
    # 6) Install Euler arrays and orientation_identifier in the phase block
    #
    # Phase entries key themselves as "id" (not "phase_id"); grains reference
    # their owning phase via "phase_id", a foreign key into that "id".
    # -------------------------------------------------------------------------
    phase_block = data.get("phase")

    if not isinstance(phase_block, list) or len(phase_block) == 0:
        raise ValueError("Missing or invalid 'phase' list in data object.")

    phase_orient = defaultdict(lambda: {"Phi1": [], "Phi": [], "Phi2": []})

    for i, grain in enumerate(grains):
        if "phase_id" not in grain:
            raise ValueError(f"Grain {i} is missing 'phase_id'.")

        phase_id = grain["phase_id"]
        orientation = grain.get("orientation")

        phase_orient[phase_id]["Phi1"].append(orientation[0])
        phase_orient[phase_id]["Phi"].append(orientation[1])
        phase_orient[phase_id]["Phi2"].append(orientation[2])

    for idx, phase_entry in enumerate(phase_block):
        if not isinstance(phase_entry, dict):
            raise ValueError(
                f"Phase entry at index {idx} is not a dict: "
                f"{type(phase_entry).__name__}"
            )

        phase_id = phase_entry.get("id", idx)

        eulers = phase_orient.get(phase_id)

        if not eulers:
            continue

        orientation_block = phase_entry.setdefault("orientation", {})
        euler_dict = orientation_block.setdefault("euler_angles", {})

        euler_dict["Phi1"] = eulers["Phi1"]
        euler_dict["Phi"] = eulers["Phi"]
        euler_dict["Phi2"] = eulers["Phi2"]

        orientation_block["grain_count"] = len(euler_dict["Phi1"])

        eulers_array = np.column_stack(
            [
                euler_dict["Phi1"],
                euler_dict["Phi"],
                euler_dict["Phi2"],
            ]
        )

        eulers_array = np.round(eulers_array, decimals=6)

        orientation_block["orientation_identifier"] = create_orientation_identifier(
            eulers_array,
            hash_length=5,
        )

    # -------------------------------------------------------------------------
    # 7) Generate deterministic MiMeDO identifier
    # -------------------------------------------------------------------------
    data["identifier"] = create_mimedo_identifier(data)
    identifier = data["identifier"]

    # -------------------------------------------------------------------------
    # 8) Build the decomposed folder key:
    #    identifier + one orientation_identifier per phase (in order) +
    #    the initial snapshot's microstructure_state_id.
    # -------------------------------------------------------------------------
    phase_orientation_ids = []

    for idx, phase_entry in enumerate(phase_block):
        orientation_id = phase_entry.get("orientation", {}).get("orientation_identifier")

        if not orientation_id:
            raise ValueError(
                f"Phase entry {idx} is missing 'orientation_identifier'; "
                "cannot build the folder key."
            )

        phase_orientation_ids.append(orientation_id)

    microstructure_state_id = initial_microstructure["microstructure_state_id"]

    key = "_".join([identifier, *phase_orientation_ids, microstructure_state_id])

    # -------------------------------------------------------------------------
    # 9) Create Keys/<key>/{inputs,results}
    # -------------------------------------------------------------------------
    case_dir = Path(base_work_dir).resolve() / key

    inputs_dir = case_dir / "inputs"
    results_dir = case_dir / "results"

    inputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    data["input_path"] = str(inputs_dir.resolve())
    data["results_path"] = str(results_dir.resolve())

    # -------------------------------------------------------------------------
    # 10) Save MiMeDO to JSON (named by identifier, not the full key -- the
    # folder path already carries the key's grouping information)
    # -------------------------------------------------------------------------
    out_path = results_dir / f"{identifier}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Saved to:", out_path.resolve())
    print("Folder key:", key)

    print("\n\n")
    print(f"{YELLOW}{OLD}MiMeDO is exported .................................................................{RESET}")
    print(f"{CYAN}{OLD}######################################################################################{RESET}")
    print("\n\n")

    return (
        data,
        str(out_path.resolve()),
        identifier,
        key,
        str(inputs_dir.resolve()),
        str(results_dir.resolve()),
    )


#############################################################################################################
#################### Writing grid, material, and load files to start a DAMASK simulation ####################
#############################################################################################################
####################################################################################################
#################### Start a new data object from an evolved microstructure ########################
####################################################################################################
@Workflow.wrap.as_function_node(
    "MiMeDO",
    "pathtoMiMeDO",
    "identifier",
    "key",
    "input_path",
    "results_path",
)
def initiate_from_dataObject(
    source: Union[str, Path],
    microstructure_state_id: str,
    user_metadata: dict,
    boundary_condition: dict,
    phase: list,
    units: dict,
    base_work_dir: Union[str, Path] = "Keys",
) -> Tuple[Dict[str, Any], str, str, str, str, str]:
    """
    Build a new MiMeDO object that starts from a microstructure an earlier run produced.

    This is the ``initialized_from`` step: it takes a finished data object
    (for example a cold-rolling run), picks the *recovered* microstructure
    inside it by identifier, and promotes that snapshot to
    ``microstructure[0]`` of a brand-new data object, which can then be
    simulated under a new load case.

    Why the snapshot cannot simply be reused in place
    -------------------------------------------------
    ``damask.GeomGrid.load_MiMedat`` always reads ``microstructure[0]``, and
    ``damask.ConfigMaterial.load_MiMedat`` always reads the *phase-level*
    ``phase[].orientation.euler_angles``. Handing a parent object straight to
    ``load_to_damask`` would therefore rebuild the parent's **initial**
    geometry and **initial** texture, silently ignoring the deformation. This
    node exists to make the evolved state the initial state, in both places.

    Selection by identifier, not by position
    ----------------------------------------
    The snapshot is named by its ``microstructure_state_id`` rather than an
    index, so the branch records exactly which microstructure it grew from
    and stays correct if snapshots are added later. The id is produced by
    ``post_processing`` (output ``recovered_state_id``) when regridding runs,
    so it can be wired straight through the workflow graph.

    Parameters
    ----------
    source : str or Path
        Path to the parent MiMeDO JSON -- the ``UpdatedDataObject`` output of
        ``post_processing`` / ``damask_run`` / ``simulate_case``.
    microstructure_state_id : str
        Identifier of the snapshot to start from, e.g. ``"S_45c6952f"``.
        Must be an ``undeformed`` (recovered/regridded) snapshot: a deformed
        grid is not a valid DAMASK grid input.
    user_metadata : dict
        Metadata for the *new* object (title, description, creator, ...).
    boundary_condition : dict
        The new load case, as ``{"mechanical_BC": [...]}``.
    phase : list of dict
        Phase definitions for the new object. Their
        ``orientation.euler_angles`` are overwritten with the evolved
        orientations taken from the selected snapshot.
    units : dict
        Unit system, as in ``write_data``.
    base_work_dir : str or Path
        Root for ``Keys/<key>/``. Default ``"Keys"``.

    Returns
    -------
    Same six outputs as ``write_data``, so the downstream nodes
    (``load_to_damask`` -> ``run_damask`` -> ``post_processing``) connect
    unchanged.

    Raises
    ------
    ValueError
        If the id is empty (regridding was disabled upstream), not found,
        ambiguous, or names a snapshot whose grid is not ``undeformed``.
    """
    print(f"{CYAN}{OLD}########################################################################################{RESET}")
    print(f"{YELLOW}{OLD}Initiating a new data object from an evolved microstructure .........................{RESET}")
    print("\n\n")

    # -------------------------------------------------------------------------
    # 1) Load the parent object
    # -------------------------------------------------------------------------
    if not microstructure_state_id:
        raise ValueError(
            "microstructure_state_id is empty. The parent run must be "
            "post-processed with do_regridding=True: only a recovered "
            "(regridded) snapshot is simulation-ready and gets an identifier."
        )

    source_path = Path(source)

    if not source_path.is_file():
        raise FileNotFoundError(f"Parent MiMeDO JSON not found: {source_path}")

    with open(source_path, "r", encoding="utf-8") as fh:
        parent = json.load(fh)

    parent_identifier = parent.get("identifier")
    snapshots = parent.get("microstructure")

    if not isinstance(snapshots, list) or len(snapshots) == 0:
        raise ValueError(f"{source_path} has no 'microstructure' list.")

    # -------------------------------------------------------------------------
    # 2) Find the requested snapshot by identifier
    # -------------------------------------------------------------------------
    matches = [
        snap for snap in snapshots
        if isinstance(snap, dict)
        and snap.get("microstructure_state_id") == microstructure_state_id
    ]

    if len(matches) == 0:
        available = [
            snap.get("microstructure_state_id")
            for snap in snapshots
            if isinstance(snap, dict) and snap.get("microstructure_state_id")
        ]
        hint = available or "none -- only regridded snapshots are given one"
        raise ValueError(
            f"No snapshot with microstructure_state_id "
            f"{microstructure_state_id!r} in {source_path}.\n"
            f"Available identifiers: {hint}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} snapshots share microstructure_state_id "
            f"{microstructure_state_id!r} in {source_path}; cannot choose."
        )

    snapshot = copy.deepcopy(matches[0])

    # -------------------------------------------------------------------------
    # 3) Refuse anything that is not simulation-ready
    #
    # A deformed grid has non-cubic, non-uniform voxels. DAMASK's grid solver
    # would still run on it, producing results that look plausible and are
    # wrong -- so this is a hard error, not a warning.
    # -------------------------------------------------------------------------
    status = snapshot.get("grid", {}).get("status")

    if status != "undeformed":
        raise ValueError(
            f"Snapshot {microstructure_state_id!r} has grid status {status!r}. "
            "Only 'undeformed' (recovered/regridded) snapshots can start a new "
            "simulation."
        )

    grains = snapshot.get("grains")
    voxels = snapshot.get("voxels")

    if not isinstance(grains, list) or len(grains) == 0:
        raise ValueError(f"Snapshot {microstructure_state_id!r} has no 'grains'.")

    if not isinstance(voxels, list) or len(voxels) == 0:
        raise ValueError(f"Snapshot {microstructure_state_id!r} has no 'voxels'.")

    # -------------------------------------------------------------------------
    # 4) Start the new object from the parent, then replace everything that
    #    describes the *old* run.
    # -------------------------------------------------------------------------
    data: Dict[str, Any] = copy.deepcopy(parent)

    # The evolved microstructure becomes the initial one, keeping its id.
    data["microstructure"] = [snapshot]

    # The parent's simulation results must not be inherited -- they belong to
    # the parent's load case, not this one.
    for results_key in ("stress", "total_strain", "plastic_strain"):
        data.pop(results_key, None)

    # New metadata, load case and units.
    data.update(copy.deepcopy(user_metadata))
    data["mechanical_BC"] = copy.deepcopy(boundary_condition["mechanical_BC"])
    data["units"] = copy.deepcopy(units)

    if not data.get("date"):
        data["date"] = datetime.utcnow().strftime("%Y-%m-%d")

    # Provenance: which object and which microstructure this grew from.
    data["initialized_from"] = {
        "identifier": parent_identifier,
        "microstructure_state_id": microstructure_state_id,
    }

    # Geometry now comes from the recovered grid, which is generally a
    # different shape from the parent's (rolling stretches the box).
    grid_block = snapshot["grid"]
    data["RVE_size"] = list(grid_block["grid_size"])
    data["discretization_unit_size"] = list(grid_block["grid_spacing"])
    data["discretization_count"] = len(voxels)

    # Software / system provenance for this object.
    data["software"] = "DAMASK"
    data["software_version"] = getattr(damask, "__version__", "unknown")
    data["system"] = platform.system()
    data["system_version"] = platform.version()
    data["processor_specifications"] = platform.processor() or platform.machine()

    # Paths are assigned below, once the key is known.
    data.pop("input_path", None)
    data.pop("results_path", None)

    # -------------------------------------------------------------------------
    # 5) Install the EVOLVED orientations into the phase block.
    #
    # ORDER IS CRITICAL. DAMASK compacts grain ids to 0..G-1 using the order
    # of the 'grains' list (`gid_to_idx = {gid: i for i, gid in
    # enumerate(gids)}` in GeomGrid.load_MiMedat), and pairs material index i
    # with Euler entry i. After segmentation the grain ids are neither
    # contiguous nor sorted, so sorting them here -- or anywhere else -- would
    # give every grain the wrong orientation, with no error raised.
    # Iterate 'grains' in list order and do not sort.
    # -------------------------------------------------------------------------
    phase_block = copy.deepcopy(phase)

    if not isinstance(phase_block, list) or len(phase_block) == 0:
        raise ValueError("'phase' must be a non-empty list of phase dicts.")

    phase_orient = defaultdict(lambda: {"Phi1": [], "Phi": [], "Phi2": []})

    for i, grain in enumerate(grains):
        if "phase_id" not in grain:
            raise ValueError(f"Grain {i} in the snapshot is missing 'phase_id'.")

        orientation = grain.get("orientation")

        if not (isinstance(orientation, (list, tuple)) and len(orientation) == 3):
            raise ValueError(
                f"Grain {i} has invalid 'orientation'. "
                f"Expected [Phi1, Phi, Phi2], got: {orientation}"
            )

        bucket = phase_orient[grain["phase_id"]]
        bucket["Phi1"].append(orientation[0])
        bucket["Phi"].append(orientation[1])
        bucket["Phi2"].append(orientation[2])

    for idx, phase_entry in enumerate(phase_block):
        if not isinstance(phase_entry, dict):
            raise ValueError(
                f"Phase entry at index {idx} is not a dict: "
                f"{type(phase_entry).__name__}"
            )

        phase_id = phase_entry.get("id", phase_entry.get("phase_id", idx))
        eulers = phase_orient.get(phase_id)

        if not eulers:
            raise ValueError(
                f"Phase entry {idx} (phase_id={phase_id}) has no grains in the "
                f"selected snapshot. Snapshot phase ids: "
                f"{sorted(phase_orient.keys())}"
            )

        orientation_block = phase_entry.setdefault("orientation", {})
        euler_dict = orientation_block.setdefault("euler_angles", {})

        euler_dict["Phi1"] = eulers["Phi1"]
        euler_dict["Phi"] = eulers["Phi"]
        euler_dict["Phi2"] = eulers["Phi2"]

        orientation_block["grain_count"] = len(euler_dict["Phi1"])

        eulers_array = np.round(
            np.column_stack(
                [euler_dict["Phi1"], euler_dict["Phi"], euler_dict["Phi2"]]
            ),
            decimals=6,
        )

        orientation_block["orientation_identifier"] = create_orientation_identifier(
            eulers_array,
            hash_length=5,
        )

    data["phase"] = phase_block

    # -------------------------------------------------------------------------
    # 6) Identifier for the new object
    # -------------------------------------------------------------------------
    data.pop("identifier", None)
    identifier = create_mimedo_identifier(data)
    data["identifier"] = identifier

    # -------------------------------------------------------------------------
    # 7) Folder key: identifier + orientation ids + the snapshot's state id
    # -------------------------------------------------------------------------
    phase_orientation_ids = []

    for idx, phase_entry in enumerate(phase_block):
        orientation_id = phase_entry.get("orientation", {}).get("orientation_identifier")

        if not orientation_id:
            raise ValueError(
                f"Phase entry {idx} is missing 'orientation_identifier'; "
                "cannot build the folder key."
            )

        phase_orientation_ids.append(orientation_id)

    key = "_".join([identifier, *phase_orientation_ids, microstructure_state_id])

    # -------------------------------------------------------------------------
    # 8) Create Keys/<key>/{inputs,results} and save
    # -------------------------------------------------------------------------
    case_dir = Path(base_work_dir).resolve() / key

    inputs_dir = case_dir / "inputs"
    results_dir = case_dir / "results"

    inputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    data["input_path"] = str(inputs_dir.resolve())
    data["results_path"] = str(results_dir.resolve())

    out_path = results_dir / f"{identifier}.json"

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    print(f"{GREEN}{OLD}Initialized from: {parent_identifier} / "
          f"{microstructure_state_id}{RESET}")
    print("Grains carried over:", len(grains))
    print("Voxels:            ", len(voxels))
    print("RVE size (um):     ",
          [round(x * 1e6, 3) for x in data["RVE_size"]])
    print("Saved to:", out_path.resolve())
    print("Folder key:", key)

    print("\n\n")
    print(f"{YELLOW}{OLD}New data object is ready ...........................................................{RESET}")
    print(f"{CYAN}{OLD}######################################################################################{RESET}")
    print("\n\n")

    return (
        data,
        str(out_path.resolve()),
        identifier,
        key,
        str(inputs_dir.resolve()),
        str(results_dir.resolve()),
    )


@Workflow.wrap.as_function_node("grid", "material", "load")
def load_to_damask(
    source: dict,
    grid_out: Union[str, Path] = "grid.vti",
    material_out: Union[str, Path] = "material.yaml",
    load_out: Union[str, Path] = "load.yaml",
    f_out_list: Sequence[int] = (5,),        # write results every f_out increments
    f_restart_list: Sequence[int] = (50,),   # write restart every f_restart increments
    outputs: Sequence[str] = ("F", "P", "F_p", "F_e", "L_p", "O"),
) -> Tuple[str, str, str]:
    """
    Build DAMASK grid, material, and load files from a MiMeDO data object.

    This function takes the MiMeDO produced by ``write_data`` and generates
    the three input files required to run a DAMASK grid/spectral simulation:

    - grid file: ``GeomGrid``
    - material file: ``ConfigMaterial``
    - loadcase file: ``LoadcaseGrid``

    The files are written to ``source["input_path"]`` (the ``inputs/``
    folder ``write_data`` already created inside ``Keys/<key>/``).

    Parameters
    ----------
    source : dict
        MiMeDO data object containing at least:

        - ``"input_path"`` : str
          Folder for DAMASK input files (created by ``write_data``).

        - ``"phase"`` : list of dicts
          Phase definitions used by ``ConfigMaterial.load_MiMedat``.

        - ``"mechanical_BC"`` : list of dicts
          Mechanical boundary-condition definitions used by
          ``LoadcaseGrid.from_mechanical_bc``. Each entry selected as the
          full-RVE FFT/spectral boundary condition (``len(vertex_list) ==
          8``) must have, per ``applied_load`` entry: ``magnitude``,
          ``duration``, and ``step`` (plus ``frequency``/``R`` if
          ``loading_mode`` is ``"cyclic"``).

    grid_out : str or pathlib.Path, optional
        Grid file name, relative to ``source["input_path"]``.
        Default is ``"grid.vti"``.

    material_out : str or pathlib.Path, optional
        Material configuration file name, relative to ``source["input_path"]``.
        Default is ``"material.yaml"``.

    load_out : str or pathlib.Path, optional
        Loadcase file name, relative to ``source["input_path"]``.
        Default is ``"load.yaml"``.

    f_out_list : sequence of int, optional
        DAMASK output frequency per load step.

    f_restart_list : sequence of int, optional
        DAMASK restart frequency per load step.

    outputs : sequence of str, optional
        Mechanical outputs requested in the material file, for example:

            ("F", "P", "F_p", "F_e", "L_p", "O")

    Returns
    -------
    tuple of str
        Absolute paths ``(grid_path, material_path, load_path)`` of the
        generated grid, material, and load files.
    """

    print(f"{CYAN}{OLD}########################################################################################{RESET}")
    print(f"{YELLOW}{OLD}Building grid/material/load from JSON ................................................{RESET}")
    print("\n\n")

    # -------------------------------------------------------------------------
    # 0) Validate inputs and resolve branch-specific input path
    # -------------------------------------------------------------------------
    if not isinstance(source, dict):
        raise ValueError("source must be a MiMeDO data object as a dict.")

    if "input_path" not in source or not source["input_path"]:
        raise ValueError(
            "source must contain a non-empty 'input_path'. "
            "This path is created by write_data and points to the branch input folder."
        )

    if not isinstance(f_out_list, Sequence) or isinstance(f_out_list, (str, bytes)):
        raise ValueError("f_out_list must be a sequence of integers.")

    if not isinstance(f_restart_list, Sequence) or isinstance(f_restart_list, (str, bytes)):
        raise ValueError("f_restart_list must be a sequence of integers.")

    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        raise ValueError("outputs must be a sequence of DAMASK output field names.")

    base_path = Path(source["input_path"]).resolve()
    base_path.mkdir(parents=True, exist_ok=True)

    grid_out = base_path / Path(grid_out).name
    material_out = base_path / Path(material_out).name
    load_out = base_path / Path(load_out).name

    # -------------------------------------------------------------------------
    # 1) Grid: build and save GeomGrid
    # -------------------------------------------------------------------------
    grid = damask.GeomGrid.load_MiMedat(source)
    grid.save(str(grid_out))

    print(grid)
    print("grid file Saved to:", grid_out)
    print("\n")

    # -------------------------------------------------------------------------
    # 2) Material: build and save ConfigMaterial
    # -------------------------------------------------------------------------
    material = damask.ConfigMaterial.load_MiMedat(source, outputs)
    material.save(str(material_out))

    print("material file Saved to:", material_out)
    print("\n")

    # -------------------------------------------------------------------------
    # 3) Loadcase: build and save LoadcaseGrid
    # -------------------------------------------------------------------------
    loadcase = damask.LoadcaseGrid.from_mechanical_bc(
        src=source,
        f_out_list=f_out_list,
        f_restart_list=f_restart_list,
        mechanical_solver="spectral_basic",
    )
    loadcase.save(str(load_out))

    print("load file Saved to:", load_out)
    print("\n")

    grid_path = str(grid_out.resolve())
    material_path = str(material_out.resolve())
    load_path = str(load_out.resolve())

    print("\n\n")
    print(f"{YELLOW}{OLD}Files are written ..................................................................{RESET}")
    print(f"{CYAN}{OLD}######################################################################################{RESET}")
    print("\n\n")

    return grid_path, material_path, load_path


#########################################################################################################
#################### Running DAMASK simulations using the FFT grid (spectral) solver ####################
#########################################################################################################
@Workflow.wrap.as_function_node("results")
def run_damask(
    grid_file: Union[str, Path],
    material_file: Union[str, Path],
    load_file: Union[str, Path],
    results_path: Union[str, Path],
    n_threads: int = 15,
) -> str:
    """
    Run a DAMASK grid/spectral simulation.

    Parameters
    ----------
    grid_file : str or pathlib.Path
        Path to the DAMASK grid file, usually ``grid.vti``.
    material_file : str or pathlib.Path
        Path to the DAMASK material file, usually ``material.yaml``.
    load_file : str or pathlib.Path
        Path to the DAMASK loadcase file, usually ``load.yaml``.
    results_path : str or pathlib.Path
        Branch-specific output directory where DAMASK results and logs are
        stored. This should be ``source["results_path"]`` from the MiMeDO
        (the ``results/`` folder ``write_data`` already created inside
        ``Keys/<key>/``).
    n_threads : int, optional
        Number of OpenMP threads used by ``DAMASK_grid`` (read via the
        ``OMP_NUM_THREADS`` environment variable).

    Returns
    -------
    str
        Absolute path to the produced DAMASK HDF5 result file.
    """

    print(f"{CYAN}{OLD}#########################################################################################{RESET}")
    print(f"{YELLOW}{OLD}Starting Simulation ...................................................................{RESET}")
    print("\n\n")

    # -------------------------------------------------------------------------
    # 1) Resolve and validate input paths
    # -------------------------------------------------------------------------
    fn_geometry = Path(grid_file).resolve()
    fn_material = Path(material_file).resolve()
    fn_load = Path(load_file).resolve()

    if not fn_geometry.is_file():
        raise FileNotFoundError(f"Geometry file not found: {fn_geometry}")

    if not fn_material.is_file():
        raise FileNotFoundError(f"Material file not found: {fn_material}")

    if not fn_load.is_file():
        raise FileNotFoundError(f"Load file not found: {fn_load}")

    if not isinstance(n_threads, int) or n_threads < 1:
        raise ValueError(f"n_threads must be an integer >= 1, got {n_threads}")

    # -------------------------------------------------------------------------
    # 2) Resolve branch-specific DAMASK output directory
    # -------------------------------------------------------------------------
    damask_dir = Path(results_path).resolve()
    damask_dir.mkdir(parents=True, exist_ok=True)

    log_path = damask_dir / "run.log"

    # -------------------------------------------------------------------------
    # 3) Build DAMASK command
    # -------------------------------------------------------------------------
    cmd = [
        "DAMASK_grid",
        "-g", str(fn_geometry),
        "-l", str(fn_load),
        "-m", str(fn_material),
    ]

    print("Running:", " ".join(cmd))
    print(f"OMP_NUM_THREADS={n_threads}")
    print("DAMASK working directory:", damask_dir)
    print("Log file:", log_path)

    # -------------------------------------------------------------------------
    # 4) Prepare environment
    # -------------------------------------------------------------------------
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_threads)

    # -------------------------------------------------------------------------
    # 5) Launch DAMASK process and write log
    # -------------------------------------------------------------------------
    with open(log_path, "w", encoding="utf-8") as log:
        print("Running:", " ".join(cmd), file=log)
        print(f"OMP_NUM_THREADS={n_threads}", file=log)
        print(f"Working directory: {damask_dir}", file=log)
        print("\n", file=log)

        process = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            cwd=str(damask_dir),
            env=env,
        )

    # -------------------------------------------------------------------------
    # 6) Check DAMASK process status
    # -------------------------------------------------------------------------
    if process.returncode != 0:
        print(f"{RED}{OLD}Error during simulation, non-zero return code{RESET}")
        raise RuntimeError(
            f"DAMASK_grid failed with return code {process.returncode}. "
            f"Check log: {log_path}"
        )

    print(f"{GREEN}{OLD}Simulation completed successfully{RESET}")

    # -------------------------------------------------------------------------
    # 7) Resolve expected HDF5 output path
    # -------------------------------------------------------------------------
    out_name = f"{fn_geometry.stem}_{fn_load.stem}_{fn_material.stem}.hdf5"
    fn_hdf = damask_dir / out_name

    if not fn_hdf.is_file():
        raise FileNotFoundError(
            f"Expected DAMASK result file not found: {fn_hdf}"
        )

    print("DAMASK result file:", fn_hdf.resolve())

    print("\n\n")
    print(f"{YELLOW}{OLD}Simulation is completed ..............................................................{RESET}")
    print(f"{CYAN}{OLD}########################################################################################{RESET}")
    print("\n\n")

    return str(fn_hdf.resolve())


#########################################################
#################### Post-processing ####################
#########################################################
##########################################################################################
#################### Stamp a recovered (regridded) snapshot with an id ####################
##########################################################################################
def _microstructure_state_id(snapshot: Dict[str, Any]) -> str:
    """
    Compute a `microstructure_state_id` for one snapshot, using Kanapy.

    Kanapy's ``create_microstructure_identifier`` is currently an *instance*
    method whose body never uses ``self`` (verified in Kanapy's ``api.py``),
    and post-processing only ever has the JSON -- there is no Microstructure
    instance to call it on. Rather than construct a dummy one, it is called
    unbound.

    The signature is inspected first so this keeps working unchanged if Kanapy
    later makes it a ``@staticmethod``. Kanapy and DAMASK are co-evolved here,
    and a silent signature change is exactly the failure mode this project has
    been bitten by before.
    """
    fn = knpy.Microstructure.create_microstructure_identifier
    first_param = next(iter(inspect.signature(fn).parameters), None)

    if first_param == "self":
        return fn(None, snapshot)
    return fn(snapshot)


def _stamp_recovered_snapshot(json_path: Union[str, Path]) -> str:
    """
    Give the last snapshot of a MiMeDO file a `microstructure_state_id`.

    Called after `append_regridded_snapshot`, whose appended snapshot is the
    recovered, simulation-ready microstructure: a regular grid again, but
    carrying the grains and orientations the deformation produced.

    Returns
    -------
    str
        The assigned identifier, e.g. ``"S_1a2b3c4d"``.

    Raises
    ------
    ValueError
        If the snapshot is not marked ``undeformed``. Kanapy's identifier
        function is documented for undeformed, simulation-ready snapshots
        only, and branching from a deformed grid would produce a DAMASK run
        that succeeds while simulating the wrong thing.
    """
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    snapshots = data.get("microstructure")
    if not isinstance(snapshots, list) or len(snapshots) == 0:
        raise ValueError(
            f"{json_path} has no 'microstructure' list to stamp."
        )

    snapshot = snapshots[-1]

    status = snapshot.get("grid", {}).get("status")
    if status != "undeformed":
        raise ValueError(
            "Refusing to stamp a microstructure_state_id on a snapshot with "
            f"grid status {status!r}. Only recovered/regridded snapshots "
            "(status 'undeformed') are simulation-ready and may be used to "
            "start a new run."
        )

    state_id = _microstructure_state_id(snapshot)
    snapshot["microstructure_state_id"] = state_id

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)

    print(f"{GREEN}{OLD}Recovered snapshot stamped: "
          f"microstructure_state_id = {state_id}{RESET}")

    return state_id


@Workflow.wrap.as_function_node("UpdatedDataObject", "recovered_state_id")
def post_processing(
    results: Union[str, Path],
    json_path: Union[str, Path],
    results_path: Union[str, Path],
    quantities: Sequence[str] = ("O", "F", "sigma", "epsilon_V^0.0(F)"),
    export_microstructure: Union[bool, str] = True,
    export_mechanical_response: bool = True,
    microstructure_stride: int = 2,
    export_vtk: bool = True,

    # segmentation / grain tracking / IPF color flags
    do_segmentation: bool = True,
    do_grain_tracking: bool = True,
    add_ipf_colors: bool = True,

    # regridding flags
    do_regridding: bool = False,
    regrid_snapshot_index: int = -1,
    regrid_spacing_sigfig: int = 1,
    regrid_out_path: Union[str, Path, None] = None,
    regrid_verbose: bool = True,
) -> Tuple[str, str]:
    """
    Post-process a DAMASK HDF5 result and update the MiMeDO JSON data object.

    This node can export:
    - homogenized mechanical response,
    - voxel-resolved microstructure snapshots,
    - segmentation,
    - grain ID tracking,
    - IPF colors,
    - optional regridding and remapping.

    Parameters
    ----------
    results : str or pathlib.Path
        Path to the DAMASK HDF5 result file.
    json_path : str or pathlib.Path
        Path to the MiMeDO JSON created by ``write_data``.
    results_path : str or pathlib.Path
        Branch-specific output directory. This should be
        ``source["results_path"]`` from the MiMeDO.
    quantities : sequence of str
        DAMASK quantities to export into the MiMeDO.
    export_microstructure : bool or str
        Controls how ``damask.Result.export_mimedo`` exports voxel-resolved
        microstructure snapshots. Accepts:

        - ``False`` -- do not export microstructure snapshots.
        - ``"voxels_only"`` -- strip grain-level data ('grains' and voxel
          'grain_id') from deformed snapshots, since the solver never
          updates those fields. Use this only when `do_segmentation=False`:
          ``segment_microstructure``'s ARI-vs-old quality metric reads
          ``voxel["grain_id"]`` unconditionally on every snapshot, including
          deformed ones, and will raise ``KeyError`` if that field is
          missing.
        - ``"voxels_with_grains"`` (also selected by passing ``True``,
          the default here) -- copies the initial grain dictionary into
          deformed snapshots too. This is explicitly not physically
          meaningful on its own (a ``UserWarning`` is raised each time) --
          it is the required *input* state for segmentation to run and
          compute its ARI-vs-old metric, and is immediately superseded by
          real grain-level data once `do_segmentation=True` runs. With the
          default settings here (``export_microstructure=True`` and
          `do_segmentation=True`), the warning is expected and harmless:
          segmentation's fresh, BFS-derived grain labels always replace the
          copied placeholder data moments later.
    export_mechanical_response : bool
        If True, export homogenized mechanical response.
    microstructure_stride : int
        Snapshot stride used during microstructure export.
    export_vtk : bool
        If True, export DAMASK VTK files.
    do_segmentation : bool
        If True, segment exported microstructure snapshots.
    do_grain_tracking : bool
        If True, track grain IDs across snapshots.
    add_ipf_colors : bool
        If True, add IPF colors for X, Y, and Z directions.
    do_regridding : bool
        If True, append a regridded snapshot.
    regrid_snapshot_index : int
        Snapshot index used for regridding.
    regrid_spacing_sigfig : int
        Significant digits used for regridding spacing.
    regrid_out_path : str or pathlib.Path or None
        Optional output path for the regridded JSON. If None, the helper decides.
    regrid_verbose : bool
        Verbosity flag for regridding helpers.

    Returns
    -------
    str
        Absolute path to the final updated JSON file.
    """

    print(f"{CYAN}{OLD}########################################################################################{RESET}")
    print(f"{YELLOW}{OLD}Start postprocessing .................................................................{RESET}")
    print("\n\n")

    # -------------------------------------------------------------------------
    # 0) Resolve paths and validate inputs
    # -------------------------------------------------------------------------
    results = Path(results).resolve()
    json_path = Path(json_path).resolve()
    out_dir = Path(results_path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results.is_file():
        raise FileNotFoundError(f"Result file not found: {results}")

    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    if not isinstance(quantities, Sequence) or isinstance(quantities, (str, bytes)):
        raise ValueError("quantities must be a sequence of DAMASK result field names.")

    if not isinstance(microstructure_stride, int) or microstructure_stride < 1:
        raise ValueError(
            f"microstructure_stride must be an integer >= 1, got {microstructure_stride}"
        )

    export_microstructure_enabled = export_microstructure is not False

    if not export_microstructure_enabled and not export_mechanical_response:
        raise ValueError(
            "At least one export branch must be enabled: "
            "export_microstructure (not False) and/or export_mechanical_response=True."
        )

    if do_regridding and not export_microstructure_enabled:
        raise ValueError(
            "do_regridding=True requires export_microstructure to be enabled "
            "(not False), because regridding needs voxel-resolved microstructure "
            "snapshots."
        )

    if do_regridding and "F" not in quantities:
        raise ValueError(
            "do_regridding=True requires 'F' in quantities, "
            "because update_grid uses deformation gradients."
        )

    if do_segmentation and export_microstructure == "voxels_only":
        raise ValueError(
            "export_microstructure='voxels_only' strips voxel['grain_id'] from "
            "deformed snapshots, but do_segmentation=True requires it to compute "
            "the ARI-vs-old quality metric. Use export_microstructure=True/"
            "'voxels_with_grains' together with do_segmentation=True, or set "
            "do_segmentation=False if you intend to use 'voxels_only'."
        )

    if regrid_out_path is not None:
        regrid_out_path = Path(regrid_out_path)

        if not regrid_out_path.is_absolute():
            regrid_out_path = out_dir / regrid_out_path

        regrid_out_path = regrid_out_path.resolve()
        regrid_out_path.parent.mkdir(parents=True, exist_ok=True)

    print("DAMASK result:", results)
    print("Input MiMeDO:", json_path)
    print("Results folder:", out_dir)
    print("\n")

    # -------------------------------------------------------------------------
    # 1) Load DAMASK result
    # -------------------------------------------------------------------------
    result = damask.Result(str(results))

    # -------------------------------------------------------------------------
    # 2) Add derived fields
    # -------------------------------------------------------------------------
    try:
        result.add_stress_Cauchy("P", "F")                 # -> sigma
        result.add_strain("F")                             # -> epsilon_V^0.0(F)
        result.add_strain("F_p", t="U", m=0.0)             # -> epsilon_U^0.0(F_p)
        result.add_equivalent_Mises("epsilon_U^0.0(F_p)")  # -> epsilon_U^0.0(F_p)_vM
        result.add_equivalent_Mises("sigma")               # -> sigma_vM
        result.add_equivalent_Mises("epsilon_V^0.0(F)")    # -> epsilon_V^0.0(F)_vM
    except ValueError as e:
        print(f"[WARN] Skipping derived-variable addition: {e}")

    # -------------------------------------------------------------------------
    # 3) Optional VTK export
    # -------------------------------------------------------------------------
    if export_vtk:
        vtk_dir = out_dir / "vtk"
        vtk_dir.mkdir(parents=True, exist_ok=True)

        try:
            result.export_VTK(target_dir=str(vtk_dir))
            print("[OK] Exported VTK to:", vtk_dir)
        except Exception as e:
            print(f"[WARN] VTK export skipped: {e}")
    else:
        print("[INFO] VTK export is disabled.")

    # -------------------------------------------------------------------------
    # 4) Export MiMeDO JSON from DAMASK result
    # -------------------------------------------------------------------------
    updated_json = result.export_mimedo(
        json_path=json_path,
        quantities=quantities,
        export_microstructure=export_microstructure,
        export_mechanical_response=export_mechanical_response,
        stride=microstructure_stride,
        verbose=True,
    )

    updated_json = Path(updated_json).resolve()

    if not updated_json.is_file():
        raise FileNotFoundError(
            f"export_mimedo did not produce a valid JSON file: {updated_json}"
        )

    print("[OK] Exported JSON to:", updated_json)
    print("\n")

    # -------------------------------------------------------------------------
    # 5) Optional segmentation / grain tracking / IPF colors
    # -------------------------------------------------------------------------
    if export_microstructure_enabled:

        if do_segmentation:
            segment_microstructure(
                updated_json,
                write_mode="inplace",
                max_angle_deg=4.0,
                connectivity=6,
                gs_min=10.0,
                include_reference=True,
                progress=True,
                verbose=True,
            )

            print("\n")
            print("[OK] Update JSON, segmentation is done")
            print("\n")
        else:
            print("[INFO] Segmentation is disabled.")

        if do_grain_tracking:
            knpy.texture.track_all_grains_across_snapshots(
                json_path=updated_json,
                log_path=str(out_dir / "track_grain_id_process.log"),
                progress=True,
                verbose=True,
            )

            print("\n")
            print("[OK] Update JSON, grain ids are updated")
            print("\n")
        else:
            print("[INFO] Grain ID tracking is disabled.")

        if add_ipf_colors:
            knpy.texture.add_ipf_color(json_path=updated_json, l=(1, 0, 0))
            knpy.texture.add_ipf_color(json_path=updated_json, l=(0, 1, 0))
            knpy.texture.add_ipf_color(json_path=updated_json, l=(0, 0, 1))

            print("\n")
            print("[OK] IPF colors are added")
            print("\n")
        else:
            print("[INFO] IPF color assignment is disabled.")

    else:
        print("[INFO] Microstructure export is disabled.")
        print("[INFO] Skipping segmentation, grain tracking, and IPF color assignment.")

    # -------------------------------------------------------------------------
    # 6) Optional regridding and mapping
    # -------------------------------------------------------------------------
    if do_regridding:
        print(f"{CYAN}{OLD}###################################################################################################{RESET}")
        print(f"{YELLOW}{OLD}Start regridding and mapping ....................................................................{RESET}")
        print("\n\n")

        # 6.1) Compute new grid proposal and average deformation gradient
        grid_size_new, grid_spacing_new, voxel_counts_new, F_avg = (
            knpy.core.rve_stats.update_grid(
                updated_json,
                snapshot_index=regrid_snapshot_index,
                spacing_sigfig=regrid_spacing_sigfig,
            )
        )

        # 6.2) Build mapping from new regular grid to old deformed voxels
        idx, box = knpy.core.rve_stats.regrid(
            json_path=updated_json,
            F_average=F_avg,
            new_grid_cell=voxel_counts_new,
            snapshot_index=regrid_snapshot_index,
            spacing_sigfig=regrid_spacing_sigfig,
            verbose=regrid_verbose,
        )

        # 6.3) Append regridded snapshot to the MiMeDO
        updated_json = knpy.core.rve_stats.append_regridded_snapshot(
            json_path=updated_json,
            idx=idx,
            grid_size_new=grid_size_new,
            grid_spacing_new=grid_spacing_new,
            voxel_counts_new=voxel_counts_new,
            snapshot_index=regrid_snapshot_index,
            out_path=regrid_out_path,
            verbose=regrid_verbose,
        )

        updated_json = Path(updated_json).resolve()

        if not updated_json.is_file():
            raise FileNotFoundError(
                f"append_regridded_snapshot did not produce a valid JSON file: {updated_json}"
            )

        # 6.4) Stamp the recovered (regridded) snapshot with a
        #      microstructure_state_id.
        #
        # append_regridded_snapshot deep-copies the *deformed* snapshot, which
        # carries no identifier, and does not assign one. Without this step no
        # snapshot except index 0 is identifiable, so a later branch has no way
        # to say which microstructure it started from.
        #
        # Kanapy's create_microstructure_identifier hashes only the basic
        # entities -- grid, grains (grain_id, phase_id, grain_volume,
        # orientation) and voxels (voxel_id, grain_id, centroid, index, volume,
        # orientation). Per-voxel simulation results (stress, strain,
        # deformation_gradient, IPF colours) are deliberately NOT part of the
        # hash, so the id identifies the microstructure itself and not the
        # loading that produced it.
        recovered_state_id = _stamp_recovered_snapshot(updated_json)

        print("\n\n")
        print(f"{YELLOW}{OLD}Mapped to JSON ..................................................................................{RESET}")
        print(f"{CYAN}{OLD}###################################################################################################{RESET}")
        print("\n\n")

    else:
        print("[INFO] Regridding is disabled.")
        # No recovered snapshot exists, so there is nothing to branch from.
        # An empty string (rather than None) keeps the output type simple; the
        # branching node rejects it with an explicit message.
        recovered_state_id = ""

    print("\n\n")
    print(f"{YELLOW}{OLD}Postprocessing is completed ....................................................................{RESET}")
    print(f"{CYAN}{OLD}##################################################################################################{RESET}")
    print("\n\n")

    return str(updated_json.resolve()), recovered_state_id
