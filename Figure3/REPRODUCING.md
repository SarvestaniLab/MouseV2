# Reproducing Figure 3F/G

## Environment

Use Python 3.11 or later. The release was validated on Windows with Python
3.13.2 and the exact direct dependency versions in `requirements.txt`.

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, replace the activation command with:

```sh
source .venv/bin/activate
```

## Selected manuscript analysis

Run:

```powershell
python refine_panels_fg.py
```

This fixes both analysis choices to the selected historical workflow:

- registration: preserved manual legacy registration;
- plane fitting: historical iterative MATLAB equations.

Expected outputs are:

```text
output/figure3_fg_full_legacy_refined.png
output/figure3_fg_full_legacy_refined.svg
output/figure3_fg_full_legacy_refined_summary.json
output/figure3_fg_full_legacy_refined_manifest.json
```

The manifest records the input-table SHA-256 and output hashes. PNG and SVG
hashes can differ across operating systems because font discovery and rendering
can differ; the numerical regression tests are the portable result check.

## Maintained corrected analysis

To run the maintained censored-normal maximum-likelihood fit on the same legacy
registration table:

```powershell
python generate_panels_fg.py `
  --registration legacy `
  --fit-method tobit `
  --output output/figure3_fg_tobit.png `
  --summary output/figure3_fg_tobit_summary.json
```

This is a secondary analysis and does not silently replace the historical
calculation used by the selected manuscript panel.

## Regression tests

```powershell
python -m unittest discover -s tests -v
```

The tests verify the checked-in table hash and row count, analysis-area
selection, historical coefficients, maintained-fit coefficients within a
cross-version numerical tolerance, and the historical within-bound sample
counts.

## Rebuilding the table

The figure commands above use the bundled compact table. To independently
rebuild that table from the public Han Figshare archive and pinned public
Zhuang-map files, follow [SOURCE_RECONSTRUCTION.md](SOURCE_RECONSTRUCTION.md).
The resulting CSV is expected to match the bundled file byte-for-byte.
