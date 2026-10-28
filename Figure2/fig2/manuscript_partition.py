"""Frozen Figure 2 partition recipe; no search or ranking dependencies."""
import numpy as np
from elastictopy.legacy_plotting import garrett_area_parcellation

def normalize_visual(full,valid):
    points=full[:,1:][valid]
    lo,hi=np.min(points,axis=0),np.max(points,axis=0)
    if np.any(hi-lo<1e-6):raise ValueError('Degenerate coordinate extent')
    return (full-(lo+hi)/2)/(hi-lo)*np.array([80.,50.])

def parcellate(data, surface, forced_row_cut=None, assignment_method="nearest_core", assignment_sign_threshold=None, merge_overlap_threshold=0.10, redundancy_threshold=1.01, post_cut_split=True):
    from scripts.make_figure2_updated import cortical_coordinate_units_per_mm
    return garrett_area_parcellation(
        data["full"],
        field_sign_sigma_mm=0.10,
        field_sign_threshold=0.25,
        surface_units_per_mm=cortical_coordinate_units_per_mm(),
        min_patch_area_mm2=0.01,
        min_patch_nodes=3,
        cleanup_radius_mm=0.03,
        dilation_radius_mm=0.15,
        eccentricity_sigma_mm=0.15,
        eccentricity_bin_step_deg=5.0,
        visual_space_pixel_size_deg=0.5,
        visual_space_close_iterations=15,
        redundancy_threshold=redundancy_threshold,
        post_cut_split=post_cut_split,
        merge_overlap_threshold=merge_overlap_threshold,
        max_split_passes=8,
        cyclic_along=True,
        surface=surface,
        v1_interior_field_sign=-1.0,
        v1_border_field_sign=0.0,
        forced_row_cut=forced_row_cut,
        assignment_method=assignment_method,
        assignment_sign_threshold=assignment_sign_threshold,
    )
