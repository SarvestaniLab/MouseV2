# Figure 3F/G legend and methods

## Figure legend

**F–G, Spatial- and temporal-frequency tuning before and after accounting for
estimated retinotopic position.** Reanalysis of Han et al. (2022), using the
preserved manual registration to the azimuth and elevation maps of Zhuang et
al. (2017). **F**, preferred spatial frequency; **G**, preferred temporal
frequency. Left, cell-level tuning distributions in six higher visual areas,
plotted on logarithmic frequency axes. Middle, fitted frequency as a function
of elevation and eccentricity from the focea, defined here as the Euclidean
distance in azimuth–elevation coordinates from (0°, +20°). Small white points
indicate estimated retinotopic positions of cells included in the fit; large
white points indicate the mean positions of the cells displayed in each area's
violin. Color indicates the back-transformed fitted frequency. Right,
cell-level residuals in octaves, calculated as observed minus fitted log2
frequency; the dashed line marks zero. Positive residuals indicate higher
preferred frequencies than predicted by retinotopic position. Gray points show
individual mouse means; large white points and bars show the equally weighted
mean across mice and its t-based 95% confidence interval in octave coordinates.
Numbers beneath each area indicate contributing mice. Plane fits pool
individual cells, including values classified as censored. The historical
analysis uses bounds of 0.02–0.32 cycles/degree for spatial frequency and
0.5–8 Hz for temporal frequency.

## Retinotopic registration and cell selection

We reanalyzed the Han et al. (2022) dataset using the reliable-cell flags and
fitted tuning parameters supplied with the data. Two-photon cell coordinates
were aligned to each animal's widefield map and then to a common cortical map
using the preserved manual registration parameters. The digitized azimuth and
elevation maps of Zhuang et al. (2017) were aligned to this common map using the
historical affine transformation. Each cell was assigned the atlas azimuth and
elevation at its registered cortical location. These coordinates are
registration-based estimates rather than independently measured single-cell
receptive-field centers.

Following the historical workflow, azimuth was clipped to 0–100° and elevation
to −40–40°. Eccentricity was calculated as
`sqrt(azimuth² + (elevation − 20)²)`, using the focea at azimuth 0° and
elevation +20° as the reference point.

The analysis included cells assigned to LM, AL, RL, AM, PM, or LI with finite
tuning and registered retinotopic coordinates. V1 and the combined POR/P
population were excluded. The retained sample comprised 15,254 cells for each
plane fit. The registration table was preserved without modification.

## Historical plane fitting and residuals

Spatial and temporal frequencies were expressed in octaves as
`log2(SF / 0.02 cycles/degree)` and `log2(TF / 0.5 Hz)`, respectively. For each
metric, a plane `mu = a * eccentricity + b * elevation + c` was fit across the
pooled cells using the historical iterative imputation and least-squares routine
preserved in `legacy_matlab/EMcheck.m`.

Values at or below 0.01 octaves and at or above 3.99 octaves were classified as
lower- and upper-censored using nominal bounds of 0 and 4 octaves. The routine
alternated its historical Gaussian-tail imputation equations with least-squares
estimation of the plane and an update of the residual scale, stopping when the
scale changed by less than 10⁻⁶ or after 100 iterations. All cells had equal
weight in the fit. The nominal bounds correspond to 0.02–0.32 cycles/degree and
0.5–8 Hz. The temporal-frequency bound is the historical analysis bound, not
the maximum tested frequency; Han et al. tested through 16 Hz.

For the violins, mouse summaries, and mean retinotopic-position markers, cells
strictly between 0.01 and 3.99 octaves were retained for the corresponding
metric. This yielded 12,105 spatial-frequency observations and 10,291
temporal-frequency observations. Residuals were calculated as observed minus
fitted octave frequency. The fitted surfaces were transformed back to physical
units using `0.02 * 2^mu` cycles/degree and `0.5 * 2^mu` Hz.

## Distribution summaries and uncertainty

Violin outlines show Gaussian kernel-density estimates of pooled cell values in
octave coordinates. Within each area, values were first averaged across cells
from the same mouse. The displayed summary is the unweighted mean of these
mouse means; its 95% confidence interval is the t-based interval across mouse
means. The contributing mouse counts are 10, 8, 8, 6, 6, and 9 for LM, AL, RL,
AM, PM, and LI, respectively.

These intervals summarize between-mouse variation conditional on the fitted
plane. They do not propagate registration or plane-fitting uncertainty. See
`KNOWN_LIMITATIONS.md` for additional interpretation constraints and for the
distinction between this historical fit and the maintained corrected Tobit fit.
