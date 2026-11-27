# Included inputs

- `training/training_inputs.npz`: fixed cortical surface, visual-field prototypes
  and V1 boundary for the selected run.
- `training/v1_boundary_cortex.npy`: 160 original V1 cortical perimeter samples
  in SVG coordinates, before the selected cyclic offset. These are geometry
  inputs, not learned RF values; their exact rounding matters during training.
- `fit_sources/hva_maps.npz`: initialization and selected training checkpoints.
- `fit_sources/expanded_footprint.svg`: selected cortical outline.
- `data/figure2_data.npz`: matched V1 and surrounding-cortex maps, cortical grids,
  partition, injection memberships and display geometry.
- `data/panel_B_partition_field.npz`: field-sign map matching the saved partition.
- `data/figure2_shared_model.json`: panel parameters and model metadata.
- `ribbon_local/ribbon_smoothing_variants.npz` and `cumulative_height_audit.json`:
  frozen illustrative registration and scale for panel C.

The package includes the inputs and reference outputs needed to reproduce
Figure 2. File checksums are recorded in `package_manifest.json`; model and
rendering parameters are in `REPRODUCTION_RECIPE.json`. Raw-to-derived
preparation and redistribution permissions remain part of the publication review.
