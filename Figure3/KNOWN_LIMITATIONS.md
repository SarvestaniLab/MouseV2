# Known limitations

- This package reproduces manuscript Figure 3 panels F and G only.
- The primary command preserves the historical MATLAB fitting equations for
  fidelity to the selected manuscript figure. Those equations are not the
  corrected censored-normal likelihood implemented by the optional Tobit mode.
- The historical temporal-frequency upper bound is 8 Hz even though the Han et
  al. experiment tested frequencies through 16 Hz. The maintained Tobit mode
  uses the complete 0.5–16 Hz range and therefore changes the selected sample
  and fitted result.
- Cell retinotopic coordinates are estimates obtained through a multistage
  manual registration, not independently measured single-cell receptive-field
  centers.
- The fitted planes pool cells. The displayed intervals summarize variation
  among mouse means conditional on the fitted plane; they do not propagate
  plane-fitting or registration uncertainty.
- The figure is descriptive. It does not by itself establish a statistically
  significant elimination of areal tuning differences after retinotopic
  subtraction.
- Rendering can vary slightly with operating system and installed fonts. The
  numerical regression tests, input hash, and output summary are the portable
  reproducibility checks.
- The public-source reconstruction covers the selected historical F/G table.
  Auxiliary registration-diagnostic workflows require `Cellrets.mat` and other
  assets that are not part of the public Figshare archive.
- The parent public repository should adopt an explicit software license before
  release; the source-data attribution terms documented in `DATA.md` are
  separate from the license applied to this analysis code.
