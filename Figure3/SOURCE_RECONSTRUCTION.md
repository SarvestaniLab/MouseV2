# Reconstructing the bundled table from public sources

The checked-in Figure 3 table can be regenerated using only public files. The
completed audit obtained an exact byte-for-byte match to
`data/han2022_figure3fg_legacy_registration.csv`.

Run these commands from the `Figure3` directory after installing
`requirements.txt`.

## 1. Download the required Han archive members

The downloader uses HTTP range requests, so it retrieves only the five needed
members of the larger Figshare ZIP. Each file is checked against the hashes in
the bundled provenance manifest.

```powershell
python source_reconstruction/fetch_public_han_inputs.py `
  --output-root work/source `
  --provenance data/provenance_legacy_registration.json
```

## 2. Obtain the public Zhuang map arrays

The two arrays are pinned to the public `add-figure2` commit used by the audit.

```powershell
git clone --depth 1 --branch add-figure2 `
  https://github.com/SarvestaniLab/MouseV2.git work/public_mousev2
git -C work/public_mousev2 fetch --depth 1 `
  origin 8d8c61dc34b7798058510a7f0661a49fc7350255
git -C work/public_mousev2 checkout --detach `
  8d8c61dc34b7798058510a7f0661a49fc7350255
```

## 3. Reconstruct and compare

```powershell
python source_reconstruction/extract_han2022_data.py `
  --source-root work/source `
  --zhuang-root work/public_mousev2/Figure2/data/zhuang2017 `
  --output work/reconstructed.csv

python source_reconstruction/compare_reconstruction.py `
  work/reconstructed.csv `
  data/han2022_figure3fg_legacy_registration.csv `
  --output work/comparison.json
```

Expected results:

- 26,046 rows;
- 23,499 rows with finite registered retinotopy;
- CSV SHA-256
  `e9704ac3c81f977bb02370ede617e59c456afdc0f4f8f0316a2a6e1eb5eb7247`;
- `exact_file_match: true` and zero maximum numeric delta in every column.

Downloaded source data and generated files belong under `work/`, which the
package `.gitignore` excludes.
