"""Verify the shipped Figure 2 files without scientific dependencies."""

from reproduce import MANIFEST_PATH, load_json, verify_package


def main() -> None:
    verify_package()
    manifest = load_json(MANIFEST_PATH)
    print(f"Verified {len(manifest['files'])} files for seed {manifest['seed']}.")


if __name__ == "__main__":
    main()
