"""Analysis and plotting for manuscript Figure 3 panels F and G.

The historical MATLAB analysis worked in octave coordinates:

* spatial frequency: ``log2(peak cycles/degree / 0.02)``
* temporal frequency: ``log2(peak Hz / 0.5)``

The maintained analysis fits a censored-normal (Tobit) plane by maximum
likelihood. Spatial frequency is censored at 0 and 4 octaves; temporal
frequency is censored at 0 and 5 octaves, corresponding to the tested
0.5--16 Hz range. The historical ``EMcheck.m`` calculation remains available
as an explicit legacy mode for exact reproduction of the submitted analysis.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t


AREA_ORDER: Tuple[str, ...] = ("LM", "AL", "RL", "AM", "PM", "LI")
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "area",
    "animal_id",
    "eccentricity_deg",
    "elevation_deg",
    "spatial_frequency_octaves",
    "temporal_frequency_octaves",
)
METRICS: Mapping[str, Mapping[str, object]] = {
    "spatial_frequency_octaves": {
        "panel": "F",
        "title": "Spatial frequency",
        "base": 0.02,
        "unit": "cycles/degree",
        "lower_bound": 0.0,
        "upper_bound": 4.0,
    },
    "temporal_frequency_octaves": {
        "panel": "G",
        "title": "Temporal frequency",
        "base": 0.5,
        "unit": "Hz",
        "lower_bound": 0.0,
        "upper_bound": 5.0,
    },
}


@dataclass(frozen=True)
class PlaneFit:
    """A fitted plane in octave coordinates."""

    eccentricity_coefficient: float
    elevation_coefficient: float
    intercept: float
    sigma: float
    iterations: int

    def predict(self, eccentricity, elevation):
        return (
            self.eccentricity_coefficient * np.asarray(eccentricity)
            + self.elevation_coefficient * np.asarray(elevation)
            + self.intercept
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "eccentricity_coefficient": self.eccentricity_coefficient,
            "elevation_coefficient": self.elevation_coefficient,
            "intercept": self.intercept,
            "sigma": self.sigma,
            "iterations": self.iterations,
        }


@dataclass(frozen=True)
class MetricAnalysis:
    """Plane fit plus the cells used for tuning and residual violins."""

    metric: str
    fit_method: str
    lower_bound: float
    upper_bound: float
    fit: PlaneFit
    all_cells: pd.DataFrame
    uncensored_cells: pd.DataFrame


def load_cells(path: Path) -> pd.DataFrame:
    """Load and validate the compact Figure 3F/G cell table."""

    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError("Figure 3 input is missing columns: " + ", ".join(missing))

    frame = frame.copy()
    frame["area"] = frame["area"].astype(str).str.strip().str.upper()
    for column in REQUIRED_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def fit_legacy_censored_plane(
    eccentricity,
    elevation,
    values,
    lower_bound: float = 0.0,
    upper_bound: float = 4.0,
    censor_margin: float = 0.01,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> PlaneFit:
    """Port the submitted MATLAB ``EMcheck.m`` plane fit.

    Censoring is detected at ``lower_bound + censor_margin`` and
    ``upper_bound - censor_margin``. The E-step intentionally matches the
    historical equations rather than replacing them with a newly derived
    censored-normal likelihood.
    """

    x = np.asarray(eccentricity, dtype=float).reshape(-1)
    y = np.asarray(elevation, dtype=float).reshape(-1)
    z = np.asarray(values, dtype=float).reshape(-1)
    if not (x.size == y.size == z.size):
        raise ValueError("eccentricity, elevation, and values must have equal length")
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[finite], y[finite], z[finite]
    if x.size < 3:
        raise ValueError("At least three finite observations are required")

    censored_below = z <= lower_bound + censor_margin
    censored_above = z >= upper_bound - censor_margin
    coefficients = np.zeros(3, dtype=float)
    sigma = 1.0
    design = np.column_stack((x, y, np.ones_like(x)))
    tiny = np.finfo(float).tiny

    for iteration in range(1, max_iterations + 1):
        mu = design @ coefficients
        imputed = z.copy()

        if np.any(censored_below):
            density = norm.pdf(
                lower_bound, loc=mu[censored_below], scale=sigma
            )
            probability = norm.cdf(
                lower_bound, loc=mu[censored_below], scale=sigma
            )
            denominator = np.maximum(1.0 - probability, tiny)
            imputed[censored_below] = (
                mu[censored_below] - sigma * density / denominator
            )

        if np.any(censored_above):
            density = norm.pdf(
                upper_bound, loc=mu[censored_above], scale=sigma
            )
            probability = norm.cdf(
                upper_bound, loc=mu[censored_above], scale=sigma
            )
            denominator = np.maximum(probability, tiny)
            imputed[censored_above] = (
                mu[censored_above] + sigma * density / denominator
            )

        coefficients = np.linalg.lstsq(design, imputed, rcond=None)[0]
        residuals = imputed - design @ coefficients
        sigma_new = float(np.sqrt(np.mean(residuals**2)))
        if not np.isfinite(sigma_new):
            raise FloatingPointError("Censored-plane fit produced a non-finite sigma")
        if abs(sigma - sigma_new) < tolerance:
            sigma = sigma_new
            break
        sigma = max(sigma_new, np.finfo(float).eps)

    return PlaneFit(
        eccentricity_coefficient=float(coefficients[0]),
        elevation_coefficient=float(coefficients[1]),
        intercept=float(coefficients[2]),
        sigma=float(sigma),
        iterations=iteration,
    )


def fit_censored_plane(
    eccentricity,
    elevation,
    values,
    lower_bound: float = 0.0,
    upper_bound: float = 4.0,
    censor_margin: float = 0.01,
    max_iterations: int = 1000,
    tolerance: float = 1e-9,
) -> PlaneFit:
    """Fit a censored-normal plane by maximum likelihood.

    Values within ``censor_margin`` of a bound contribute the appropriate
    lower- or upper-tail probability. Values between the bounds contribute
    their normal density. Predictors are standardized during optimization for
    numerical stability and coefficients are returned in degree units.
    """

    if not lower_bound < upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound")

    x = np.asarray(eccentricity, dtype=float).reshape(-1)
    y = np.asarray(elevation, dtype=float).reshape(-1)
    z = np.asarray(values, dtype=float).reshape(-1)
    if not (x.size == y.size == z.size):
        raise ValueError("eccentricity, elevation, and values must have equal length")
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[finite], y[finite], z[finite]
    if x.size < 3:
        raise ValueError("At least three finite observations are required")

    means = np.asarray((np.mean(x), np.mean(y)), dtype=float)
    scales = np.asarray((np.std(x), np.std(y)), dtype=float)
    if np.any(scales <= np.finfo(float).eps):
        raise ValueError("Both retinotopic predictors must vary")
    standardized = np.column_stack(((x - means[0]) / scales[0],
                                    (y - means[1]) / scales[1]))
    design = np.column_stack((standardized, np.ones_like(x)))

    censored_below = z <= lower_bound + censor_margin
    censored_above = z >= upper_bound - censor_margin
    uncensored = ~(censored_below | censored_above)
    if np.sum(uncensored) < 3:
        raise ValueError("At least three uncensored observations are required")

    initial_coefficients = np.linalg.lstsq(
        design[uncensored], z[uncensored], rcond=None
    )[0]
    initial_residuals = z[uncensored] - design[uncensored] @ initial_coefficients
    initial_sigma = max(float(np.std(initial_residuals, ddof=0)), 0.1)
    initial = np.concatenate((initial_coefficients, (np.log(initial_sigma),)))

    def negative_log_likelihood(parameters):
        coefficients = parameters[:3]
        sigma = np.exp(parameters[3])
        mu = design @ coefficients
        log_likelihood = 0.0
        if np.any(uncensored):
            log_likelihood += np.sum(
                norm.logpdf(z[uncensored], loc=mu[uncensored], scale=sigma)
            )
        if np.any(censored_below):
            standardized_lower = (
                lower_bound - mu[censored_below]
            ) / sigma
            log_likelihood += np.sum(norm.logcdf(standardized_lower))
        if np.any(censored_above):
            standardized_upper = (
                upper_bound - mu[censored_above]
            ) / sigma
            log_likelihood += np.sum(norm.logsf(standardized_upper))
        return -float(log_likelihood)

    result = minimize(
        negative_log_likelihood,
        initial,
        method="L-BFGS-B",
        bounds=((None, None), (None, None), (None, None),
                (np.log(np.finfo(float).eps), np.log(1e6))),
        options={"maxiter": max_iterations, "ftol": tolerance},
    )
    if not result.success:
        raise RuntimeError("Censored-normal fit failed: " + result.message)

    standardized_coefficients = result.x[:3]
    coefficients = standardized_coefficients[:2] / scales
    intercept = (
        standardized_coefficients[2]
        - np.sum(standardized_coefficients[:2] * means / scales)
    )
    return PlaneFit(
        eccentricity_coefficient=float(coefficients[0]),
        elevation_coefficient=float(coefficients[1]),
        intercept=float(intercept),
        sigma=float(np.exp(result.x[3])),
        iterations=int(result.nit),
    )


def analyze_metric(
    frame: pd.DataFrame, metric: str, fit_method: str = "tobit"
) -> MetricAnalysis:
    """Fit one tuning metric and calculate cell-wise octave residuals."""

    if metric not in METRICS:
        raise ValueError("Unknown Figure 3 metric: " + metric)
    columns = ["area", "animal_id", "eccentricity_deg", "elevation_deg", metric]
    cells = frame.loc[frame["area"].isin(AREA_ORDER), columns].dropna().copy()
    missing_areas = [area for area in AREA_ORDER if area not in set(cells["area"])]
    if missing_areas:
        raise ValueError("Figure 3 input has no cells for: " + ", ".join(missing_areas))

    if fit_method == "tobit":
        lower_bound = float(METRICS[metric]["lower_bound"])
        upper_bound = float(METRICS[metric]["upper_bound"])
        fit = fit_censored_plane(
            cells["eccentricity_deg"], cells["elevation_deg"], cells[metric],
            lower_bound=lower_bound, upper_bound=upper_bound,
        )
    elif fit_method == "legacy":
        lower_bound = 0.0
        upper_bound = 4.0
        fit = fit_legacy_censored_plane(
            cells["eccentricity_deg"], cells["elevation_deg"], cells[metric],
            lower_bound=lower_bound, upper_bound=upper_bound,
        )
    else:
        raise ValueError("fit_method must be 'tobit' or 'legacy'")
    cells["predicted_octaves"] = fit.predict(
        cells["eccentricity_deg"], cells["elevation_deg"]
    )
    cells["residual_octaves"] = cells[metric] - cells["predicted_octaves"]
    uncensored = cells.loc[
        (cells[metric] > lower_bound + 0.01)
        & (cells[metric] < upper_bound - 0.01)
    ].copy()
    return MetricAnalysis(
        metric, fit_method, lower_bound, upper_bound, fit, cells, uncensored
    )


def _area_values(frame: pd.DataFrame, column: str) -> Sequence[np.ndarray]:
    return [
        frame.loc[frame["area"] == area, column].to_numpy(dtype=float)
        for area in AREA_ORDER
    ]


def _mouse_area_values(
    frame: pd.DataFrame, area: str, column: str
) -> np.ndarray:
    """Return one equally weighted value per animal for an area."""

    return (
        frame.loc[frame["area"] == area]
        .groupby("animal_id", sort=True)[column]
        .mean()
        .to_numpy(dtype=float)
    )


def _draw_violin(
    axis, frame: pd.DataFrame, column: str, ylabel: str = ""
) -> None:
    values = _area_values(frame, column)
    positions = np.arange(1, len(values) + 1)
    violins = axis.violinplot(
        values, positions=positions, widths=0.86,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body in violins["bodies"]:
        body.set_facecolor("black")
        body.set_edgecolor("black")
        body.set_alpha(1.0)

    mouse_values = [
        _mouse_area_values(frame, area, column) for area in AREA_ORDER
    ]
    means = np.asarray([np.mean(value) for value in mouse_values])
    ci95 = np.asarray([
        student_t.ppf(0.975, value.size - 1)
        * np.std(value, ddof=1) / np.sqrt(value.size)
        if value.size > 1 else 0.0
        for value in mouse_values
    ])
    for position, per_mouse in zip(positions, mouse_values):
        offsets = (
            np.linspace(-0.20, 0.20, per_mouse.size)
            if per_mouse.size > 1 else np.zeros(1)
        )
        axis.scatter(
            position + offsets, per_mouse, s=10, facecolor="0.72",
            edgecolor="black", linewidth=0.35, zorder=3,
        )

    lower, upper = means - ci95, means + ci95
    cap_width = 0.075
    axis.vlines(positions, lower, upper, color="black", linewidth=2.6, zorder=4)
    axis.vlines(positions, lower, upper, color="white", linewidth=1.2, zorder=5)
    axis.hlines(lower, positions - cap_width, positions + cap_width,
                color="black", linewidth=2.6, zorder=4)
    axis.hlines(upper, positions - cap_width, positions + cap_width,
                color="black", linewidth=2.6, zorder=4)
    axis.hlines(lower, positions - cap_width, positions + cap_width,
                color="white", linewidth=1.2, zorder=5)
    axis.hlines(upper, positions - cap_width, positions + cap_width,
                color="white", linewidth=1.2, zorder=5)
    axis.scatter(
        positions, means, s=22, color="white", edgecolor="black",
        linewidth=0.65, zorder=6,
    )
    tick_labels = [
        f"{area}\nn={per_mouse.size}"
        for area, per_mouse in zip(AREA_ORDER, mouse_values)
    ]
    axis.set_xticks(positions, tick_labels)
    axis.tick_params(axis="x", labelsize=8, pad=2)
    axis.set_xlim(0.45, len(values) + 0.55)
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)


def _physical_tick_labels(
    base: float, upper_bound: float
) -> Tuple[np.ndarray, Sequence[str]]:
    octaves = np.arange(0.0, upper_bound + 1.0)
    physical = base * 2.0**octaves
    labels = [f"{value:g}" for value in physical]
    return octaves, labels


def _plot_metric_row(axes, analysis: MetricAnalysis) -> None:
    settings = METRICS[analysis.metric]
    base = float(settings["base"])
    title = str(settings["title"])
    unit = str(settings["unit"])
    panel = str(settings["panel"])

    _draw_violin(
        axes[0], analysis.uncensored_cells, analysis.metric,
        f"{title}\n({unit})",
    )
    ticks, labels = _physical_tick_labels(base, analysis.upper_bound)
    axes[0].set_yticks(ticks, labels)
    axes[0].set_ylim(analysis.lower_bound, analysis.upper_bound)
    axes[0].set_title("Tuning by area", fontsize=10)
    axes[0].text(-0.28, 1.08, panel, transform=axes[0].transAxes,
                 fontsize=17, fontweight="bold", va="top")

    cells = analysis.all_cells
    x_min, x_max = np.nanpercentile(cells["eccentricity_deg"], (0.5, 99.5))
    y_min, y_max = np.nanpercentile(cells["elevation_deg"], (0.5, 99.5))
    x_grid = np.linspace(x_min, x_max, 180)
    y_grid = np.linspace(y_min, y_max, 140)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    predicted = base * 2.0 ** analysis.fit.predict(x_mesh, y_mesh)
    image = axes[1].imshow(
        predicted, origin="lower", aspect="auto",
        extent=(x_min, x_max, y_min, y_max), cmap="viridis",
    )
    axes[1].scatter(
        cells["eccentricity_deg"], cells["elevation_deg"],
        s=0.15, color="white", alpha=0.28, linewidths=0,
    )
    for area in AREA_ORDER:
        selected = analysis.uncensored_cells.loc[
            analysis.uncensored_cells["area"] == area
        ]
        x_mean = float(selected["eccentricity_deg"].mean())
        y_mean = float(selected["elevation_deg"].mean())
        axes[1].scatter(x_mean, y_mean, s=28, facecolor="white", edgecolor="black", zorder=4)
        if area == "AM":
            label_x, alignment = x_mean - 1.2, "right"
        else:
            label_x, alignment = x_mean + 1.2, "left"
        axes[1].text(
            label_x, y_mean + 0.8, area, fontsize=8, color="black",
            ha=alignment, zorder=5,
        )
    axes[1].set_xlabel("Eccentricity (degrees from fovea)")
    axes[1].set_ylabel("Elevation (degrees)")
    axes[1].set_title("Plane fit", fontsize=10)
    colorbar = axes[1].figure.colorbar(image, ax=axes[1], fraction=0.045, pad=0.03)
    colorbar.set_label(unit, fontsize=8)

    _draw_violin(axes[2], analysis.uncensored_cells, "residual_octaves", "")
    axes[2].set_ylim(-2.0, 2.0)
    axes[2].set_yticklabels([])
    axes[2].set_title("Residuals", fontsize=10)


def plot_panels_fg(
    frame: pd.DataFrame, output_path: Path, fit_method: str = "tobit"
):
    """Render the two-row F/G panel group and return its analyses."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analyses = {
        metric: analyze_metric(frame, metric, fit_method=fit_method)
        for metric in METRICS
    }
    figure, axes = plt.subplots(
        2, 3, figsize=(10.5, 6.1),
        gridspec_kw={"width_ratios": (1.0, 2.15, 1.0)},
        constrained_layout=True,
    )
    for row, metric in enumerate(METRICS):
        _plot_metric_row(axes[row], analyses[metric])
    figure.patch.set_facecolor("white")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, facecolor="white")
    plt.close(figure)
    return analyses


def write_analysis_summary(
    analyses: Mapping[str, MetricAnalysis], output_path: Path
) -> None:
    """Write coefficients and cell counts used by a rendered panel group."""

    payload = {}
    for metric, analysis in analyses.items():
        payload[metric] = {
            "fit_method": analysis.fit_method,
            "lower_censoring_bound_octaves": analysis.lower_bound,
            "upper_censoring_bound_octaves": analysis.upper_bound,
            "fit": analysis.fit.as_dict(),
            "all_cell_count": int(len(analysis.all_cells)),
            "uncensored_cell_count": int(len(analysis.uncensored_cells)),
            "uncensored_cell_count_by_area": {
                area: int(np.sum(analysis.uncensored_cells["area"] == area))
                for area in AREA_ORDER
            },
            "mouse_count_by_area": {
                area: int(analysis.uncensored_cells.loc[
                    analysis.uncensored_cells["area"] == area, "animal_id"
                ].nunique())
                for area in AREA_ORDER
            },
        }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
