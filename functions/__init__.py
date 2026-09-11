"""
Shared, standardized building blocks for the microstructure workflows.

Everything reusable lives here; `../cases/` holds only the case notebooks
and their case-specific data.

    nodes                  -- the pyiron_workflow pipeline nodes
    mimedo_identifier      -- deterministic identifiers for MiMeDO objects
    mimedo_viewer          -- interactive / static viewer for a MiMeDO JSON
    segment_microstructure -- misorientation-based grain segmentation
    voxel_graph            -- voxel neighbour graph (used by mimedo_viewer)
"""
