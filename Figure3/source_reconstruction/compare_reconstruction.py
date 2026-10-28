"""Compare a public-only reconstruction with the checked-in Figure 3 table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reconstructed", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    new = pd.read_csv(args.reconstructed)
    reference = pd.read_csv(args.reference)
    report: dict[str, object] = {
        "reconstructed_sha256": sha256(args.reconstructed),
        "reference_sha256": sha256(args.reference),
        "exact_file_match": args.reconstructed.read_bytes() == args.reference.read_bytes(),
        "reconstructed_rows": len(new),
        "reference_rows": len(reference),
        "columns_match": list(new.columns) == list(reference.columns),
    }
    if report["columns_match"] and len(new) == len(reference):
        deltas = {}
        for column in new.select_dtypes(include=np.number).columns:
            left = new[column].to_numpy(float)
            right = reference[column].to_numpy(float)
            finite = np.isfinite(left) & np.isfinite(right)
            deltas[column] = {
                "nan_pattern_match": bool(np.array_equal(np.isnan(left), np.isnan(right))),
                "max_abs_delta": float(np.max(np.abs(left[finite] - right[finite])))
                if finite.any() else None,
            }
        report["numeric_deltas"] = deltas
        report["all_values_match"] = bool(
            all(
                np.allclose(new[column], reference[column], equal_nan=True, rtol=0, atol=0)
                if pd.api.types.is_numeric_dtype(new[column])
                else new[column].equals(reference[column])
                for column in new.columns
            )
        )

    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
