# Figure 3F/G: tuning differences after accounting for retinotopy

This directory reproduces manuscript Figure 3 panels F and G. It reanalyzes
the spatial- and temporal-frequency tuning data of Han et al. (2022) after the
preserved manual registration to the azimuth and elevation maps of Zhuang et
al. (2017).

The primary command reproduces the selected manuscript analysis: the preserved
manual registration together with the historical MATLAB plane-fitting equations.
The repository also retains a corrected censored-normal maximum-likelihood fit
as an explicitly labeled secondary analysis.

## Quick start

Python 3.11 or later is recommended. From this directory:

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python refine_panels_fg.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

The command writes PNG and SVG versions of the figure, a numerical analysis
summary, and a hash manifest beneath `output/`.

Run the regression tests with:

```powershell
python -m unittest discover -s tests -v
```

See [REPRODUCING.md](REPRODUCING.md) for the complete workflow,
[DATA.md](DATA.md) for input provenance and licensing, [METHODS.md](METHODS.md)
for the figure legend and analysis details, and
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for interpretation caveats. The
public-source derivation can be repeated with
[SOURCE_RECONSTRUCTION.md](SOURCE_RECONSTRUCTION.md); its completed verification
is recorded in [PUBLIC_DATA_AUDIT.md](PUBLIC_DATA_AUDIT.md).
The bundled table can also be rebuilt solely from public upstream files; see
[SOURCE_RECONSTRUCTION.md](SOURCE_RECONSTRUCTION.md) and the completed
[public-data audit](PUBLIC_DATA_AUDIT.md).

## Scope

This release covers panels F and G only. It includes the minimal public-source
reconstruction path for the bundled table, but intentionally omits exploratory
objective registrations, field-sign audits, per-mouse QC sheets, and other
development workflows that are not required to reproduce the selected panels.

## References

- Han X, Vermaercke B, Bonin V. Diversity of spatiotemporal coding reveals
  specialized visual processing streams in the mouse cortex. *Nature
  Communications* 13, 3249 (2022). <https://doi.org/10.1038/s41467-022-29656-z>
- Zhuang J, Ng L, Williams D, Valley M, Li Y, Garrett M, Waters J. An extended
  retinotopic map of mouse cortex. *eLife* 6:e18372 (2017).
  <https://doi.org/10.7554/eLife.18372>
