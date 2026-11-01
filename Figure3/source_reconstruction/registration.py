"""Objective registration of Zhuang retinotopy maps to the Han common map.

The spatial affine transform is estimated using Han 1p retinotopy values only.
SF and TF tuning values are deliberately absent from this module.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize


# MATLAB affine2d convention:
# [x_out, y_out] = [x_source, y_source] @ linear + translation.
LEGACY_PARAMETERS = np.asarray((-1.0, 0.5, 0.5, 1.05, 370.0, -180.0))
PARAMETER_BOUNDS: Tuple[Tuple[float, float], ...] = (
    (-1.5, -0.5),  # x -> x, including the known left/right reflection
    (0.0, 1.0),
    (0.0, 1.0),
    (0.5, 1.5),
    (250.0, 450.0),
    (-300.0, 0.0),
)


@dataclass(frozen=True)
class RegistrationFit:
    """An affine spatial transform and its training diagnostics."""

    parameters: np.ndarray
    objective: float
    iterations: int
    success: bool
    message: str

    @property
    def linear(self) -> np.ndarray:
        return np.asarray((
            (self.parameters[0], self.parameters[1]),
            (self.parameters[2], self.parameters[3]),
        ))

    @property
    def translation(self) -> np.ndarray:
        return self.parameters[4:6]

    def as_dict(self) -> Dict[str, object]:
        return {
            "parameters": [float(value) for value in self.parameters],
            "linear": self.linear.tolist(),
            "translation": self.translation.tolist(),
            "objective": float(self.objective),
            "iterations": int(self.iterations),
            "success": bool(self.success),
            "message": self.message,
        }


def unpack_affine(parameters: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (6,):
        raise ValueError("An affine transform requires six parameters")
    linear = np.asarray((
        (parameters[0], parameters[1]),
        (parameters[2], parameters[3]),
    ))
    translation = parameters[4:6]
    return linear, translation


def sample_zhuang_maps(
    common_i,
    common_j,
    altitude_source: np.ndarray,
    azimuth_source: np.ndarray,
    parameters: Sequence[float],
    interpolation_order: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample source maps at common-map positions under an affine transform."""

    common_i = np.asarray(common_i, dtype=float).reshape(-1)
    common_j = np.asarray(common_j, dtype=float).reshape(-1)
    if common_i.size != common_j.size:
        raise ValueError("common_i and common_j must have equal length")
    linear, translation = unpack_affine(parameters)
    determinant = float(np.linalg.det(linear))
    if abs(determinant) < 1e-8:
        return (
            np.full(common_i.size, np.nan),
            np.full(common_i.size, np.nan),
        )

    output_xy = np.vstack((common_j, common_i))
    source_xy = np.linalg.solve(
        linear.T, output_xy - translation.reshape(2, 1)
    )
    coordinates = np.vstack((source_xy[1] - 1.0, source_xy[0] - 1.0))
    altitude = map_coordinates(
        altitude_source, coordinates, order=interpolation_order,
        mode="constant", cval=np.nan, prefilter=False,
    )
    azimuth = map_coordinates(
        azimuth_source, coordinates, order=interpolation_order,
        mode="constant", cval=np.nan, prefilter=False,
    )
    return altitude, azimuth


def equal_mouse_weights(animal_id: Iterable[float]) -> np.ndarray:
    """Give every mouse equal total weight and every cell within it equal weight."""

    animal_id = np.asarray(animal_id).reshape(-1)
    unique, inverse, counts = np.unique(
        animal_id, return_inverse=True, return_counts=True
    )
    return 1.0 / (unique.size * counts[inverse])


def registration_weights(
    animal_id: Iterable[float], weighting: str = "equal_mouse"
) -> np.ndarray:
    """Return normalized cell weights for a registration objective."""

    animal_id = np.asarray(animal_id).reshape(-1)
    if weighting == "equal_mouse":
        return equal_mouse_weights(animal_id)
    if weighting == "equal_cell":
        return np.full(animal_id.size, 1.0 / animal_id.size)
    raise ValueError("weighting must be 'equal_mouse' or 'equal_cell'")


def aggregate_spatial_bins(
    common_i,
    common_j,
    animal_id,
    measured_azimuth,
    measured_elevation,
    bin_size_pixels: float = 10.0,
) -> Tuple[np.ndarray, ...]:
    """Collapse repeated cells to one median observation per mouse/spatial bin."""

    if bin_size_pixels <= 0:
        raise ValueError("bin_size_pixels must be positive")
    frame = pd.DataFrame({
        "common_i": np.asarray(common_i, dtype=float),
        "common_j": np.asarray(common_j, dtype=float),
        "animal_id": np.asarray(animal_id),
        "measured_azimuth": np.asarray(measured_azimuth, dtype=float),
        "measured_elevation": np.asarray(measured_elevation, dtype=float),
    }).dropna()
    frame["bin_i"] = np.floor(frame["common_i"] / bin_size_pixels)
    frame["bin_j"] = np.floor(frame["common_j"] / bin_size_pixels)
    aggregated = (
        frame.groupby(["animal_id", "bin_i", "bin_j"], sort=True)[[
            "common_i", "common_j", "measured_azimuth", "measured_elevation"
        ]]
        .median()
        .reset_index()
    )
    return tuple(
        aggregated[column].to_numpy()
        for column in (
            "common_i", "common_j", "animal_id",
            "measured_azimuth", "measured_elevation",
        )
    )


def _weighted_calibration(target, prediction, weights):
    design = np.column_stack((prediction, np.ones_like(prediction)))
    root_weight = np.sqrt(weights)
    coefficients = np.linalg.lstsq(
        design * root_weight[:, None], target * root_weight, rcond=None
    )[0]
    return coefficients


def _dimension_metrics(
    target,
    prediction,
    weights,
    calibration=None,
) -> Dict[str, object]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite_target = np.isfinite(target)
    total_weight = float(np.sum(weights[finite_target]))
    valid = finite_target & np.isfinite(prediction)
    valid_weight = float(np.sum(weights[valid]))
    if np.sum(valid) < 3 or valid_weight <= 0.0:
        return {
            "loss": 1e6,
            "valid_weight_fraction": 0.0,
            "calibration": [np.nan, np.nan],
            "weighted_r_squared": np.nan,
            "weighted_rmse": np.nan,
            "correlation": np.nan,
        }

    local_weights = weights[valid] / valid_weight
    local_target = target[valid]
    local_prediction = prediction[valid]
    if calibration is None:
        calibration = _weighted_calibration(
            local_target, local_prediction, local_weights
        )
    calibration = np.asarray(calibration, dtype=float)
    fitted = calibration[0] * local_prediction + calibration[1]
    residual = local_target - fitted
    target_mean = float(np.sum(local_weights * local_target))
    target_variance = float(np.sum(
        local_weights * (local_target - target_mean) ** 2
    ))
    mse = float(np.sum(local_weights * residual**2))
    r_squared = 1.0 - mse / max(target_variance, np.finfo(float).eps)
    correlation = float(np.corrcoef(local_target, local_prediction)[0, 1])
    valid_fraction = valid_weight / max(total_weight, np.finfo(float).eps)
    missing_penalty = 20.0 * (1.0 - valid_fraction) ** 2
    return {
        "loss": float(1.0 - r_squared + missing_penalty),
        "valid_weight_fraction": float(valid_fraction),
        "calibration": calibration.tolist(),
        "weighted_r_squared": float(r_squared),
        "weighted_rmse": float(np.sqrt(mse)),
        "correlation": correlation,
    }


def evaluate_registration(
    parameters,
    common_i,
    common_j,
    animal_id,
    measured_azimuth,
    measured_elevation,
    altitude_source,
    azimuth_source,
    calibrations: Mapping[str, Sequence[float]] = None,
    weighting: str = "equal_mouse",
) -> Dict[str, object]:
    """Evaluate retinotopy agreement with equal total weight per mouse."""

    altitude, azimuth = sample_zhuang_maps(
        common_i, common_j, altitude_source, azimuth_source, parameters
    )
    weights = registration_weights(animal_id, weighting=weighting)
    calibrations = calibrations or {}
    azimuth_metrics = _dimension_metrics(
        measured_azimuth, azimuth, weights, calibrations.get("azimuth")
    )
    elevation_metrics = _dimension_metrics(
        measured_elevation, altitude, weights, calibrations.get("elevation")
    )
    return {
        "loss": float(np.mean((
            azimuth_metrics["loss"], elevation_metrics["loss"]
        ))),
        "azimuth": azimuth_metrics,
        "elevation": elevation_metrics,
    }


def fit_objective_affine(
    common_i,
    common_j,
    animal_id,
    measured_azimuth,
    measured_elevation,
    altitude_source,
    azimuth_source,
    initial_parameters: Sequence[float] = LEGACY_PARAMETERS,
    max_iterations: int = 400,
    weighting: str = "equal_mouse",
) -> RegistrationFit:
    """Estimate the global affine without consulting SF or TF tuning."""

    common_i = np.asarray(common_i, dtype=float)
    common_j = np.asarray(common_j, dtype=float)
    animal_id = np.asarray(animal_id)
    measured_azimuth = np.asarray(measured_azimuth, dtype=float)
    measured_elevation = np.asarray(measured_elevation, dtype=float)
    finite = (
        np.isfinite(common_i) & np.isfinite(common_j)
        & np.isfinite(animal_id)
        & np.isfinite(measured_azimuth) & np.isfinite(measured_elevation)
    )
    arrays = (
        common_i[finite], common_j[finite], animal_id[finite],
        measured_azimuth[finite], measured_elevation[finite],
    )
    if np.unique(arrays[2]).size < 2:
        raise ValueError("Registration fitting requires at least two mice")

    lower = np.asarray([bound[0] for bound in PARAMETER_BOUNDS])
    upper = np.asarray([bound[1] for bound in PARAMETER_BOUNDS])

    def to_physical(unit_parameters):
        return lower + np.asarray(unit_parameters) * (upper - lower)

    def to_unit(parameters):
        return (np.asarray(parameters) - lower) / (upper - lower)

    def objective(unit_parameters):
        parameters = to_physical(unit_parameters)
        return evaluate_registration(
            parameters, *arrays, altitude_source, azimuth_source,
            weighting=weighting,
        )["loss"]

    result = minimize(
        objective,
        np.clip(to_unit(initial_parameters), 0.0, 1.0),
        method="L-BFGS-B",
        bounds=((0.0, 1.0),) * 6,
        options={"maxiter": max_iterations, "ftol": 1e-10, "maxls": 40},
    )
    parameters = to_physical(result.x)
    return RegistrationFit(
        parameters=parameters,
        objective=float(result.fun),
        iterations=int(result.nit),
        success=bool(result.success),
        message=str(result.message),
    )
