"""Download only the public Han archive members needed for Figure 3.

The 828 MB Figshare ZIP supports HTTP range requests. ``remotezip`` lets this
script retrieve the five required members without storing the whole archive.
Every file is checked against the hashes already recorded in Figure 3's
provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from remotezip import RemoteZip


ARCHIVE_URL = "https://ndownloader.figshare.com/files/49816254"
ARCHIVE_PREFIX = PurePosixPath(
    "Han 2022 NatComm data retinotopy"
)
MEMBERS = (
    "dataset_meta.mat",
    "s_retinomaps.mat",
    "s_roi_location_all_8areas.mat",
    "code_dependency/data_script_spatial_organization.m",
    "spatial organization/retino_map.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "figure3/data/provenance.json",
    )
    args = parser.parse_args()

    expected = {
        item["file"]: (item["bytes"], item["sha256"])
        for item in json.loads(args.provenance.read_text())["source_files"]
        if item["file"] in MEMBERS
    }
    destination_root = (
        args.output_root / "visualize cell location and tuning"
        / Path(*ARCHIVE_PREFIX.parts)
    )

    with RemoteZip(ARCHIVE_URL) as archive:
        names = set(archive.namelist())
        for relative_name in MEMBERS:
            archive_name = str(ARCHIVE_PREFIX / PurePosixPath(relative_name))
            if archive_name not in names:
                raise FileNotFoundError(f"Missing archive member: {archive_name}")
            destination = destination_root / Path(*PurePosixPath(relative_name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or (
                destination.stat().st_size,
                sha256(destination),
            ) != expected[relative_name]:
                with archive.open(archive_name) as source, destination.open("wb") as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
            actual = (destination.stat().st_size, sha256(destination))
            if actual != expected[relative_name]:
                raise ValueError(
                    f"Integrity check failed for {relative_name}: {actual} != "
                    f"{expected[relative_name]}"
                )
            print(f"verified {relative_name} ({actual[0]:,} bytes)")


if __name__ == "__main__":
    main()
