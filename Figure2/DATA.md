# Data availability for the publication draft

The export includes compact inputs needed by the selected entry points. Inclusion
in this review draft does not establish redistribution permission. Confirm the
licenses and attribution for all derived inputs before the final release.

| Inputs | Role and provenance | Remaining work |
| --- | --- | --- |
| `data/publication/figure2/` | Frozen model checkpoints and inputs, original run metadata with portable paths, and checksums | Verify retraining parity |
| `data/reference/MouseV1_synthetic*` | Schematic visual-field boundary and companion metadata; informed by Wagor and Garrett with Waters anchoring | Finalize source citations and distinguish synthetic assumptions from observations |
| `data/waters2019/derived/WatersV1_*` | Waters-derived cortical borders; companion JSON where available | Confirm redistribution and document raw-to-derived preparation |
| `data/zhuang2017/*.mat`, `data/ZhuangTracing9_borders.svg` | Three historical Zhuang interpolation/mask arrays and tracing used by Figure 2F | Confirm source provenance and redistribution permission |
| `data/waters2019/derived/manifest.json` | Cortical coordinate scale calibration used by Figure 2 | Retain source provenance |

The historical Zhuang arrays were obtained from the local Han/Bonin data archive;
the original Zhuang attribution and redistribution terms still need review.
Archive reference: Han, Vermaercke and Bonin (2022), *Diversity of spatiotemporal coding reveals
specialized visual processing streams in the mouse cortex*, Nature
Communications 13, 3249: https://doi.org/10.1038/s41467-022-29656-z.

Raw Waters arrays, bulk Han/Bonin source data, literature images, exploratory fits,
and large output bundles are excluded. The documented model and Figure 2 commands use the included inputs. Raw-data regeneration is a separate,
uncompleted release requirement. References to excluded files inside provenance
metadata describe historical inputs, not dependencies of those panel commands.
