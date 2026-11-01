# Public-only Han Figure 3 reconstruction audit

Audit run: 2026-10-28

## Outcome

Figure 3F/G was successfully regenerated using only files obtainable from
public sources. The reconstructed compact CSV and rendered historical figure
are both byte-for-byte identical to their reference counterparts.

| Check | Result |
|---|---:|
| Reconstructed rows | 26,046 |
| Finite registered-retinotopy rows | 23,499 |
| CSV SHA-256 | `e9704ac3c81f977bb02370ede617e59c456afdc0f4f8f0316a2a6e1eb5eb7247` |
| CSV byte comparison | Exact match |
| Maximum numeric delta, any column | 0 |
| Rendered legacy PNG SHA-256 | `9352302b1a0dcda4dec98e860064b23e9cc81bcf6023b50c007ce52430d19d5a` |
| PNG byte comparison | Exact match |

The analysis summaries also matched exactly. The legacy-fit coefficients were:

| Tuning dimension | Eccentricity | Elevation | Intercept | Sigma |
|---|---:|---:|---:|---:|
| Spatial frequency (octaves) | 0.011080628683448744 | 0.016770612774416604 | 1.4225905737886158 | 0.8775593568872725 |
| Temporal frequency (octaves) | -0.01602660545322438 | -0.029264038563212358 | 2.831226910427385 | 0.856183708104914 |

## Public sources

- Han et al. (2022) Figshare archive, file ID `49816254`, retrieved by HTTP
  range requests from `https://ndownloader.figshare.com/files/49816254`.
- Zhuang interpolated retinotopy maps from public repository
  `https://github.com/SarvestaniLab/MouseV2`, branch `add-figure2`, audited at
  commit `8d8c61dc34b7798058510a7f0661a49fc7350255`.

The Figshare archive contains the exact five Han inputs required by the
default extractor; each matched the size and SHA-256 recorded in
`figure3/data/provenance.json`.

`Cellrets.mat` is absent from the public archive. It supplies only the optional
Han one-photon azimuth/elevation columns used for registration diagnostics.
It is not read when generating the default Figure 3 table.

The public altitude MAT file's whole-file hash differs from the historical
hash in the provenance manifest, while the public azimuth MAT file matches.
MAT files can differ at the container/metadata level after being re-saved. The
public altitude file nevertheless produced an exactly identical CSV and PNG,
showing that this serialization difference does not affect the analysis.

## Interpretation

For the maintained historical Figure 3F/G path, replacing the previously held
Han source folder with the public Figshare archive changes neither the data
table, fitted model, counts, nor rendered figure. This audit does not establish
that every auxiliary registration-diagnostic workflow can run from public data:
those workflows still require `Cellrets.mat` and potentially other local assets.
