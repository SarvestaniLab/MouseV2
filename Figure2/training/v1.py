"""CPU V1 elastic-net fit on a harmonic grid with fixed perimeter RFs.

Run with Python 3.11 and requirements-training.txt. The V1 objective shares
coefficients with HVA but uses a nonperiodic grid and fixed boundary links.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def geometry(border):
    """Keep all border samples exactly; harmonically extend cortical positions."""
    if len(border) % 4:
        raise ValueError("Boundary length must be divisible by four")
    import numpy as np
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    m = len(border) // 4
    side = m + 1
    perimeter = (
        [(0, j) for j in range(m)]
        + [(i, m) for i in range(m)]
        + [(m, j) for j in range(m, 0, -1)]
        + [(i, 0) for i in range(m, 0, -1)]
    )
    grid = np.zeros((side, side, 2))
    for ij, p in zip(perimeter, border):
        grid[ij] = p
    interior = [(i, j) for i in range(1, m) for j in range(1, m)]
    lookup = {ij: k for k, ij in enumerate(interior)}
    boundary_lookup = {ij: k for k, ij in enumerate(perimeter)}
    a = lil_matrix((len(interior), len(interior)))
    rhs = np.zeros((len(interior), 2))
    for ij, k in lookup.items():
        i, j = ij
        a[k, k] = 4
        for q in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
            if q in lookup:
                a[k, lookup[q]] = -1
            else:
                rhs[k] += grid[q]
    positions = spsolve(a.tocsr(), rhs)
    for ij, p in zip(interior, positions):
        grid[ij] = p
    edges, anchors = [], []
    for (i, j), k in lookup.items():
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == dj == 0:
                    continue
                q = (i + di, j + dj)
                if q in lookup:
                    edges.append((k, lookup[q]))
                else:
                    anchors.append((k, boundary_lookup[q]))
    edges, anchors = np.array(edges), np.array(anchors)
    delta = positions[edges[:, 1]] - positions[edges[:, 0]]
    weights = 1 / np.sum(delta**2, axis=1)
    moment = np.zeros((len(interior), 2, 2))
    np.add.at(
        moment,
        edges[:, 0],
        weights[:, None, None] * delta[:, :, None] * delta[:, None, :],
    )
    assert np.all(np.linalg.matrix_rank(moment) == 2)
    coeff = np.einsum(
        "eab,eb->ea", np.linalg.inv(moment)[edges[:, 0]], weights[:, None] * delta
    )
    indices, values = [], []
    n = len(interior)
    for c in range(2):
        indices.extend(np.column_stack((c * n + edges[:, 0], edges[:, 1])).tolist())
        indices.extend(np.column_stack((c * n + np.arange(n), np.arange(n))).tolist())
        values.extend(coeff[:, c])
        values.extend(-np.bincount(edges[:, 0], weights=coeff[:, c], minlength=n))
    indices, values = np.array(indices), np.array(values)
    order = np.lexsort((indices[:, 1], indices[:, 0]))
    geo = dict(
        valid_nodes=np.ones(n, bool),
        derivative_operator_indices=indices[order],
        derivative_operator_values=values[order],
        derivative_operator_shape=np.array([2 * n, n]),
    )
    return grid, np.array(interior), np.array(perimeter), edges, weights, anchors, geo


def cell_determinants(grid):
    import numpy as np

    a, b, c, d = grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]
    cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return np.stack(
        [
            cross(b - a, d - a),
            cross(c - b, a - b),
            cross(d - c, b - c),
            cross(a - d, c - d),
        ]
    )


def fit(output_dir):
    from reproduce import sha256, validate_output_dir, verify_package

    validate_output_dir(output_dir)
    if output_dir.exists():
        raise SystemExit(
            "Choose a new output directory; existing results are preserved."
        )
    verify_package()
    if sys.version_info[:2] != (3, 11):
        raise SystemExit(
            "V1 training requires Python 3.11 and requirements-training.txt."
        )
    import importlib.metadata

    for name, version in {
        "numpy": "1.26.4",
        "scipy": "1.14.1",
        "tensorflow": "2.17.1",
    }.items():
        if importlib.metadata.version(name) != version:
            raise SystemExit(f"V1 training requires {name}=={version}.")
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        os.environ[key] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import numpy as np
    from elastictopy import elastic_net_mouse as en

    inputs = np.load(ROOT / "training/training_inputs.npz")
    config = json.loads((ROOT / "training/training_config.json").read_text())
    run, par = config["run"], config["schedule"]
    # Preserve the original sampled V1 boundary: rounding it through the HVA
    # mesh changes the optimizer trajectory despite sub-1e-14 differences.
    border = np.load(ROOT / "training/v1_boundary_cortex.npy")
    scores = []
    for shift in range(len(border) // 4):
        grid, *_ = geometry(np.roll(border, -shift, axis=0))
        determinants = cell_determinants(grid)
        scores.append(float(np.min(determinants * np.sign(np.median(determinants)))))
    shift = int(np.argmax(scores))
    grid, interior, perimeter, edges, weights, anchors, geo = geometry(
        np.roll(border, -shift, axis=0)
    )
    det = cell_determinants(grid)
    if np.min(det * np.sign(np.median(det))) <= 1e-12:
        raise ValueError("Folded cortical grid")
    boundary = np.roll(inputs["boundary"], -shift, axis=0)
    targets = inputs["targets"]
    rng = np.random.RandomState(run["seed"])
    initial = targets[rng.randint(len(targets), size=len(interior))] + rng.normal(
        0, 1, (len(interior), 2)
    )
    tf = en.tf
    tf.reset_default_graph()
    tf.set_random_seed(run["seed"])
    y = tf.Variable(initial, dtype=tf.float64)
    kappa = tf.placeholder(tf.float64, ())
    coverage, _ = en.main_cost(tf.constant(targets), y, kappa, stable_logsumexp=True)
    smooth = tf.reduce_sum(
        weights
        * tf.reduce_sum(
            tf.square(tf.gather(y, edges[:, 0]) - tf.gather(y, edges[:, 1])), axis=1
        )
    )
    boundary_cost = tf.reduce_sum(
        tf.square(tf.gather(y, anchors[:, 0]) - boundary[anchors[:, 1]])
    )
    anisotropy = en.local_metric_isotropy_energy(y, geo)
    parts = [coverage, smooth, boundary_cost, anisotropy]
    cost = (
        coverage
        + run["b1"] * smooth
        + run["b2"] * boundary_cost
        + run["b3"] * anisotropy
    )
    opt = en.generate_optimizer(
        cost, eta0=par["eta0"], m=par["momentum"], eta_decay=par["eta_decay"]
    )
    started = time.perf_counter()
    history = []
    with tf.Session(
        config=tf.ConfigProto(
            intra_op_parallelism_threads=2, inter_op_parallelism_threads=2
        )
    ) as session:
        session.run(tf.global_variables_initializer())
        k = par["kappa0"]
        for step in range(1, par["steps"] + 1):
            session.run(opt, {kappa: k})
            if step % 100 == 0 or step == par["steps"]:
                final, losses = session.run([y, parts], {kappa: k})
                if not np.isfinite(final).all() or not np.isfinite(losses).all():
                    raise ValueError("Nonfinite V1 fit")
                history.append([step, k, *map(float, losses)])
            if step % 500 == 0:
                print(f"V1: {step}/{par['steps']} updates", flush=True)
            k *= 1 - par["kappa_decay"]
    visual = np.zeros_like(grid)
    visual[perimeter[:, 0], perimeter[:, 1]] = boundary
    visual[interior[:, 0], interior[:, 1]] = final
    reference = np.load(ROOT / "data/figure2_data.npz")["rf_v1"][1:-1, 1:-1].reshape(
        -1, 2
    )
    delta = final - reference
    receipt = dict(
        seed=run["seed"],
        steps=par["steps"],
        seconds=time.perf_counter() - started,
        boundary_cyclic_offset=shift,
        endpoint_exact=bool(np.array_equal(final, reference)),
        endpoint_rms_deg=float(np.sqrt(np.mean(delta**2))),
        endpoint_max_abs_deg=float(np.max(abs(delta))),
        environment={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "tensorflow")
        },
        python=sys.version.split()[0],
        config=config,
        input_sha256={
            name: sha256(ROOT / "training" / name)
            for name in (
                "training_inputs.npz",
                "v1_boundary_cortex.npy",
                "training_config.json",
            )
        },
        status="COMPLETED",
        note="Endpoint differences are reported, not treated as evidence of exact reconstruction.",
    )
    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "fit.npz",
        cortical_grid=grid,
        visual_grid=visual,
        initial=initial,
        final=final,
        targets=targets,
        boundary=boundary,
        history=history,
    )
    (output_dir / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/v1")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    fit(args.output_dir.resolve())


if __name__ == "__main__":
    main()
