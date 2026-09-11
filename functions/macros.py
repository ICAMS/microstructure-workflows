"""
Macro nodes: reusable multi-node chunks of the pipeline.

Why macros
----------
The single-branch Cold Rolling case wires seven nodes by hand. A case with
several loading branches would repeat four of those nodes once per branch,
so the notebook becomes mostly copy-paste -- exactly the drift problem this
package exists to remove.

Two macros are defined here, and the split between them is deliberate:

    damask_run     load_to_damask -> run_damask -> post_processing
    simulate_case  write_data -> damask_run

`damask_run` is the part that is identical for *every* branch, no matter
where its microstructure came from. `simulate_case` adds the "start from a
freshly generated RVE" front end.

A later `simulate_branch` (start from an evolved microstructure produced by
a previous run) will reuse `damask_run` unchanged and only swap the front
end. Keeping the tail separate now is what makes that possible without a
rewrite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Union

import kanapy as knpy
from pyiron_workflow import Workflow

from .nodes import (
    write_data, load_to_damask, run_damask, post_processing,
    initiate_from_dataObject,
)


@Workflow.wrap.as_macro_node("UpdatedDataObject", "results", "recovered_state_id")
def damask_run(
    self,
    MiMeDO: dict,
    pathtoMiMeDO: Union[str, Path],
    results_path: Union[str, Path],

    # --- DAMASK input-file controls (load_to_damask) ---
    f_out_list: Sequence[int] = (1,),
    f_restart_list: Sequence[int] = (100,),
    damask_outputs: Sequence[str] = ("F", "P", "F_p", "F_e", "L_p", "O"),

    # --- solver controls (run_damask) ---
    n_threads: int = 10,

    # --- post-processing controls ---
    quantities: Sequence[str] = ("O", "F", "sigma", "epsilon_V^0.0(F)"),
    export_microstructure: Union[bool, str] = True,
    export_mechanical_response: bool = True,
    microstructure_stride: int = 10,
    export_vtk: bool = True,
    do_segmentation: bool = True,
    do_grain_tracking: bool = True,
    add_ipf_colors: bool = True,
    do_regridding: bool = False,
):
    """
    Run one DAMASK simulation from a finished MiMeDO object and post-process it.

    Everything downstream of "we have a MiMeDO object describing this run".
    Takes the MiMeDO produced by `write_data` (or, later, by a branching
    node) and returns the same object updated with the simulation results.

    Parameters
    ----------
    MiMeDO : dict
        The MiMeDO data object. `load_to_damask` reads the grid, material
        and load case out of it, and locates its own output folder from the
        object's `input_path`.
    pathtoMiMeDO : str or Path
        Path of the MiMeDO JSON on disk. `post_processing` writes the
        updated object back here.
    results_path : str or Path
        The run's `Keys/<key>/results/` folder -- where DAMASK writes its
        HDF5 and where post-processing puts VTK output.

    Returns
    -------
    UpdatedDataObject
        The MiMeDO object with simulation results merged in.
    results
        Path to the DAMASK HDF5 result file.
    recovered_state_id
        `microstructure_state_id` of the recovered (regridded) snapshot, so a
        later branch can name the exact microstructure it starts from. Empty
        when `do_regridding=False`, since no recovered snapshot exists.

    Notes
    -----
    `run_damask` launches the real solver. This macro is where the compute
    time of a case is spent.
    """
    self.load_to_damask = load_to_damask(
        source=MiMeDO,
        f_out_list=f_out_list,
        f_restart_list=f_restart_list,
        outputs=damask_outputs,   # node arg is `outputs`; macro input
                                  # is renamed because pyiron reserves that name
    )

    self.run_damask = run_damask(
        grid_file=self.load_to_damask.outputs.grid,
        material_file=self.load_to_damask.outputs.material,
        load_file=self.load_to_damask.outputs.load,
        results_path=results_path,
        n_threads=n_threads,
    )

    self.post_processing = post_processing(
        results=self.run_damask,
        json_path=pathtoMiMeDO,
        results_path=results_path,
        quantities=quantities,
        export_microstructure=export_microstructure,
        export_mechanical_response=export_mechanical_response,
        microstructure_stride=microstructure_stride,
        export_vtk=export_vtk,
        do_segmentation=do_segmentation,
        do_grain_tracking=do_grain_tracking,
        add_ipf_colors=add_ipf_colors,
        do_regridding=do_regridding,
    )

    return (
        self.post_processing.outputs.UpdatedDataObject,
        self.run_damask.outputs.results,
        self.post_processing.outputs.recovered_state_id,
    )


@Workflow.wrap.as_macro_node(
    "UpdatedDataObject", "results", "identifier", "key", "results_path",
    "recovered_state_id",
)
def simulate_case(
    self,
    # Must match write_data's own hint exactly -- pyiron requires a macro
    # input hint to be at least as specific as the child input it feeds.
    RVE: knpy.Microstructure,
    user_metadata: dict,
    boundary_condition: dict,
    phase: list,
    units: dict,
    base_work_dir: Union[str, Path] = "Keys",

    # --- forwarded to damask_run ---
    f_out_list: Sequence[int] = (1,),
    f_restart_list: Sequence[int] = (100,),
    damask_outputs: Sequence[str] = ("F", "P", "F_p", "F_e", "L_p", "O"),
    n_threads: int = 10,
    quantities: Sequence[str] = ("O", "F", "sigma", "epsilon_V^0.0(F)"),
    export_microstructure: Union[bool, str] = True,
    export_mechanical_response: bool = True,
    microstructure_stride: int = 10,
    export_vtk: bool = True,
    do_segmentation: bool = True,
    do_grain_tracking: bool = True,
    add_ipf_colors: bool = True,
    do_regridding: bool = False,
):
    """
    One complete simulation branch, starting from a generated RVE.

    Chains `write_data` (build the MiMeDO object and its `Keys/<key>/`
    folder) into `damask_run` (simulate and post-process).

    Several instances can share the same `RVE` input. They then produce
    MiMeDO objects with the *same* orientation_identifier and
    microstructure_state_id but a *different* mimedo_identifier -- because
    `mechanical_BC` is part of the hashed identifier fields. The result is
    one folder per branch, visibly sharing one microstructure:

        <id_A>_<orientation_id>_<microstructure_state_id>
        <id_B>_<orientation_id>_<microstructure_state_id>
                ^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^ identical

    Parameters
    ----------
    RVE : knpy.Microstructure
        The voxelized RVE, normally the output of `generate_rve`. Safe to
        share between branches: `write_data` assigns (never appends to) the
        phase Euler arrays, so repeated calls are idempotent.
    user_metadata, boundary_condition, phase, units : dict / list
        Passed straight to `write_data`. `boundary_condition` holds the two
        complementary tensors (prescribed strain, prescribed stress); every
        component must be prescribed in exactly one of them.
    base_work_dir : str or Path
        Root for the `Keys/<key>/` folders. Default "Keys".

    Returns
    -------
    UpdatedDataObject
        MiMeDO object with results merged in.
    results
        Path to the DAMASK HDF5 result file.
    identifier, key, results_path
        This branch's identity and output folder, handy for reporting and
        for later branching off this run.
    recovered_state_id
        `microstructure_state_id` of this run's recovered (regridded)
        snapshot. Wire this into a branching node to start a new simulation
        from the microstructure this run produced. Empty unless
        `do_regridding=True`.
    """
    self.write_data = write_data(
        source=RVE,
        user_metadata=user_metadata,
        boundary_condition=boundary_condition,
        phases=phase,
        units=units,
        base_work_dir=base_work_dir,
    )

    self.damask_run = damask_run(
        MiMeDO=self.write_data.outputs.MiMeDO,
        pathtoMiMeDO=self.write_data.outputs.pathtoMiMeDO,
        results_path=self.write_data.outputs.results_path,
        f_out_list=f_out_list,
        f_restart_list=f_restart_list,
        damask_outputs=damask_outputs,
        n_threads=n_threads,
        quantities=quantities,
        export_microstructure=export_microstructure,
        export_mechanical_response=export_mechanical_response,
        microstructure_stride=microstructure_stride,
        export_vtk=export_vtk,
        do_segmentation=do_segmentation,
        do_grain_tracking=do_grain_tracking,
        add_ipf_colors=add_ipf_colors,
        do_regridding=do_regridding,
    )

    return (
        self.damask_run.outputs.UpdatedDataObject,
        self.damask_run.outputs.results,
        self.write_data.outputs.identifier,
        self.write_data.outputs.key,
        self.write_data.outputs.results_path,
        self.damask_run.outputs.recovered_state_id,
    )


@Workflow.wrap.as_macro_node(
    "UpdatedDataObject", "results", "identifier", "key", "results_path",
    "recovered_state_id",
)
def simulate_branch(
    self,
    source: str,
    microstructure_state_id: str,
    user_metadata: dict,
    boundary_condition: dict,
    phase: list,
    units: dict,
    base_work_dir: Union[str, Path] = "Keys",

    # --- forwarded to damask_run ---
    f_out_list: Sequence[int] = (1,),
    f_restart_list: Sequence[int] = (100,),
    damask_outputs: Sequence[str] = ("F", "P", "F_p", "F_e", "L_p", "O"),
    n_threads: int = 10,
    quantities: Sequence[str] = ("O", "F", "sigma", "epsilon_V^0.0(F)"),
    export_microstructure: Union[bool, str] = True,
    export_mechanical_response: bool = True,
    microstructure_stride: int = 10,
    export_vtk: bool = True,
    do_segmentation: bool = True,
    do_grain_tracking: bool = True,
    add_ipf_colors: bool = True,
    do_regridding: bool = False,
):
    """
    One simulation branch that starts from a microstructure a previous run produced.

    The counterpart of `simulate_case`: same tail, different front end.

        simulate_case    write_data               -> damask_run
        simulate_branch  initiate_from_dataObject -> damask_run

    `damask_run` is reused unchanged -- which is why it was split out in the
    first place. Everything downstream of "we have a MiMeDO object" is
    identical whether that object describes a freshly generated RVE or one
    inherited from an earlier simulation.

    Parameters
    ----------
    source : str
        Path to the parent MiMeDO JSON: the `UpdatedDataObject` output of the
        run to branch from.
    microstructure_state_id : str
        Which microstructure inside that object to start from. Wire this from
        the parent's `recovered_state_id` output so the graph carries it
        automatically; the parent must have run with `do_regridding=True`.

    Returns
    -------
    Same outputs as `simulate_case`.
    """
    self.initiate_from_dataObject = initiate_from_dataObject(
        source=source,
        microstructure_state_id=microstructure_state_id,
        user_metadata=user_metadata,
        boundary_condition=boundary_condition,
        phase=phase,
        units=units,
        base_work_dir=base_work_dir,
    )

    self.damask_run = damask_run(
        MiMeDO=self.initiate_from_dataObject.outputs.MiMeDO,
        pathtoMiMeDO=self.initiate_from_dataObject.outputs.pathtoMiMeDO,
        results_path=self.initiate_from_dataObject.outputs.results_path,
        f_out_list=f_out_list,
        f_restart_list=f_restart_list,
        damask_outputs=damask_outputs,
        n_threads=n_threads,
        quantities=quantities,
        export_microstructure=export_microstructure,
        export_mechanical_response=export_mechanical_response,
        microstructure_stride=microstructure_stride,
        export_vtk=export_vtk,
        do_segmentation=do_segmentation,
        do_grain_tracking=do_grain_tracking,
        add_ipf_colors=add_ipf_colors,
        do_regridding=do_regridding,
    )

    return (
        self.damask_run.outputs.UpdatedDataObject,
        self.damask_run.outputs.results,
        self.initiate_from_dataObject.outputs.identifier,
        self.initiate_from_dataObject.outputs.key,
        self.initiate_from_dataObject.outputs.results_path,
        self.damask_run.outputs.recovered_state_id,
    )
