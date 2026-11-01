# Data provenance

The bundled `data/han2022_figure3fg_legacy_registration.csv` is a compact,
derived analysis table containing the reliable cells and variables required for
Figure 3F/G. It contains 26,046 rows. Its SHA-256 is:

```text
e9704ac3c81f977bb02370ede617e59c456afdc0f4f8f0316a2a6e1eb5eb7247
```

The table records area and animal identifiers, registered azimuth and elevation,
eccentricity from the analysis focea, and preferred spatial and temporal
frequency in octave coordinates. V1 and POR/P rows are retained for provenance
but excluded from the F/G plane fits and plots.

`data/provenance_legacy_registration.json` records source-file names, byte
sizes, SHA-256 hashes, row counts, and transformation notes. The large raw
source files are not redistributed here, but the table can be regenerated from
their public locations by following `SOURCE_RECONSTRUCTION.md`.

## Source dataset

Han X, Vermaercke B, Bonin V. Diversity of spatiotemporal coding reveals
specialized visual processing streams in the mouse cortex. *Nature
Communications* 13, 3249 (2022).
<https://doi.org/10.1038/s41467-022-29656-z>

The authors make the cell-location/tuning archive and related data available
through Figshare:
<https://doi.org/10.6084/m9.figshare.18415925>.

The Figshare item is marked Creative Commons Attribution 4.0 International:
<https://creativecommons.org/licenses/by/4.0/>. The public-data audit matched
the five Han archive members used by the default extractor to the recorded
provenance hashes and reproduced this CSV byte-for-byte. `Cellrets.mat` is not
in the public archive, but it supplies only optional registration-diagnostic
columns and is not used to generate this table.

This derived table changes the source material by selecting reliable cells,
applying the preserved coordinate registrations, clipping the historical
retinotopic ranges, and computing octave and eccentricity variables. See
`PUBLIC_DATA_AUDIT.md` for exact input and output checks.

## Retinotopic atlas

Registered azimuth and elevation values use maps from:

Zhuang J, Ng L, Williams D, Valley M, Li Y, Garrett M, Waters J. An extended
retinotopic map of mouse cortex. *eLife* 6:e18372 (2017).
<https://doi.org/10.7554/eLife.18372>

Users should cite both source studies when using this derived table or the
reproduced panels.
