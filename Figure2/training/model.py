"""Initialization and training for the single selected Figure 2 run.

Numerical operations and TensorFlow graph names are retained for replay parity.
"""

import os

os.environ.update(
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    TF_NUM_INTRAOP_THREADS="1",
    TF_NUM_INTEROP_THREADS="1",
    TF_DETERMINISTIC_OPS="1",
    TF_CPP_MIN_LOG_LEVEL="3",
)
import time
import numpy as np
from training import optimizer as bench


def seeded_initial(targets, seed):
    """Sample 2,560 prototypes with replacement and add unit Gaussian RF noise."""
    rng = np.random.RandomState(seed)
    return targets[rng.randint(len(targets), size=2560)] + rng.normal(0, 1, (2560, 2))


def build_graph(plan, surface, targets, boundary, lanes=1):
    """Build the frozen float64 objective on the supplied 160-by-17 cortical grid.

    The inner column supplies fixed boundary geometry; 160-by-16 RFs are learned.
    The geometry and GPU energy providers are restored after graph construction.
    The selected replay uses one lane.
    """
    tf = bench.tf
    schedule = plan["schedule"]
    original_generator = bench.en.generate_mouse_surface_grid
    geometry_calls = []

    def provider(svg_path=bench.en.DEFAULT_BORDER_SVG, map_h=80, map_w=8, **kwargs):
        if (map_h, map_w) != (160, 16):
            raise ValueError(f"unexpected mesh request: {(map_h, map_w)}")
        geometry_calls.append(str(svg_path))
        return surface.copy()

    bench.en.generate_mouse_surface_grid = provider
    tf.reset_default_graph()
    tf.set_random_seed(91001)
    lane_graphs = []
    try:
        for lane_index in range(int(lanes)):
            with tf.name_scope(f"sweep_lane_{lane_index}"), tf.device("/GPU:0"):
                values = tf.Variable(
                    np.zeros((2560, 2)), dtype=tf.float64, name="values"
                )
                original_energy = bench.en.local_metric_isotropy_energy
                bench.en.local_metric_isotropy_energy = bench.dense_energy
                try:
                    cost, parts, placeholders, _ = bench.en.generate_elastic_net(
                        targets,
                        boundary,
                        values,
                        160,
                        16,
                        b3_objective=bench.en.B3_LOCAL_METRIC_ISOTROPY,
                        stable_coverage=True,
                        surface_svg=plan["mesh_svg"],
                    )
                finally:
                    bench.en.local_metric_isotropy_energy = original_energy
                optimizer = bench.en.generate_optimizer(
                    cost,
                    eta0=schedule["eta0"],
                    m=schedule["momentum"],
                    eta_decay=schedule["eta_decay"],
                )
                initial_placeholder = tf.placeholder(
                    tf.float64, (2560, 2), name="sweep_initial"
                )
                assign_initial = values.assign(initial_placeholder)
            lane_graphs.append(
                dict(
                    values=values,
                    cost=cost,
                    parts=parts,
                    placeholders=placeholders,
                    optimizer=optimizer,
                    initial_placeholder=initial_placeholder,
                    assign_initial=assign_initial,
                )
            )
        initialize = tf.global_variables_initializer()
    finally:
        bench.en.generate_mouse_surface_grid = original_generator
    return dict(lanes=lane_graphs, initialize=initialize, geometry_calls=geometry_calls)


def train_one(session, graph_bundle, run, plan, initial, mode="callable", lane_index=0):
    """Reset model and RMSProp state, anneal kappa, and retain requested updates."""
    graph = graph_bundle["lanes"][lane_index]
    schedule = plan["schedule"]
    session.run(graph_bundle["initialize"])
    session.run(graph["assign_initial"], {graph["initial_placeholder"]: initial})
    snapshots = {"initial": initial}
    snapshot_steps = set(plan["snapshots"])
    kappa = schedule["kappa0"]
    values = [run["b1"], run["b2"], run["b3"], kappa]
    if mode == "callable":
        train = session.make_callable(
            graph["optimizer"], feed_list=graph["placeholders"]
        )
    elif mode == "legacy":
        feed = dict(zip(graph["placeholders"], values))
    else:
        raise ValueError(mode)
    started = time.perf_counter()
    for step in range(1, schedule["steps"] + 1):
        values[3] = kappa
        if mode == "callable":
            train(*values)
        else:
            feed[graph["placeholders"][3]] = kappa
            session.run(graph["optimizer"], feed)
        kappa -= kappa * schedule["kappa_decay"]
        if step in snapshot_steps:
            snapshots[f"step_{step}"] = session.run(graph["values"])
    seconds = time.perf_counter() - started
    final = snapshots[f"step_{schedule['steps']}"]
    if not np.isfinite(final).all():
        raise ValueError("nonfinite endpoint")
    values[3] = kappa
    final_parts = session.run(graph["parts"], dict(zip(graph["placeholders"], values)))
    snapshots["final"] = final
    return snapshots, seconds, [float(value) for value in final_parts]
