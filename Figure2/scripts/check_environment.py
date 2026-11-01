"""Deterministic smoke check for the Python simulation environment."""

from __future__ import print_function

import hashlib
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import scipy
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "elastictopy"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

EXPECTED_VISUAL_FIELD_SHAPE = (256, 2)
# Current pooled Waters density default, c=-0.05. The prior digest
# 1a2a562751383bdea30679b217f3f094928afc35122d2df5ec86ddc77e8942c2
# belongs to c=-0.5 and is not the current runner's coverage baseline.
EXPECTED_VISUAL_FIELD_SHA256 = (
    "7ce7829d38dbf91fc71151a14b76a62292aa8bee7ffc612275b6178db3e4e668"
)
EXPECTED_BOUNDARY_SHAPE = (80, 2)
EXPECTED_BOUNDARY_SHA256 = (
    "b15d593fed6eff44079e430755f5d5e959a31517dc218b21fcd8148908925468"
)

from run_mouse_v2 import make_shared_inputs


def fingerprint(array):
    """Return a stable digest for a numeric array's shape, dtype, and values."""
    contiguous = np.ascontiguousarray(array)
    payload = repr((contiguous.shape, contiguous.dtype.str)).encode("ascii")
    payload += contiguous.tobytes()
    return hashlib.sha256(payload).hexdigest()


def main():
    visual_field, boundary, _ = make_shared_inputs(
        seed=20260812,
        map_h=80,
        map_w=8,
        prototype_count=256,
    )

    tf.compat.v1.disable_eager_execution()
    graph = tf.Graph()
    with graph.as_default():
        value = tf.reduce_sum(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
    with tf.compat.v1.Session(graph=graph) as session:
        tensorflow_sum = float(session.run(value))

    visual_field_digest = fingerprint(visual_field)
    boundary_digest = fingerprint(boundary)

    print("python={}.{}.{}".format(*sys.version_info[:3]))
    print("tensorflow={}".format(tf.__version__))
    print("numpy={}".format(np.__version__))
    print("scipy={}".format(scipy.__version__))
    print("pandas={}".format(pd.__version__))
    print("visual_field_shape={}".format(visual_field.shape))
    print("visual_field_sha256={}".format(visual_field_digest))
    print("boundary_shape={}".format(boundary.shape))
    print("boundary_sha256={}".format(boundary_digest))
    print("tensorflow_sum={:.1f}".format(tensorflow_sum))

    if tensorflow_sum != 10.0:
        raise RuntimeError("TensorFlow smoke calculation returned an unexpected result")
    if visual_field.shape != EXPECTED_VISUAL_FIELD_SHAPE:
        raise RuntimeError("Visual-field sample has an unexpected shape")
    if visual_field_digest != EXPECTED_VISUAL_FIELD_SHA256:
        raise RuntimeError("Visual-field sample does not match the coverage baseline")
    if boundary.shape != EXPECTED_BOUNDARY_SHAPE:
        raise RuntimeError("V1 boundary has an unexpected shape")
    if boundary_digest != EXPECTED_BOUNDARY_SHA256:
        raise RuntimeError("V1 boundary does not match the coverage baseline")


if __name__ == "__main__":
    main()
