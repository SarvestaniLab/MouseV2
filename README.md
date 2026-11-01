# MouseV2: a single complex V2 in mouse visual cortex

Data, analysis code, and figure reproductions for:

> Rowley DP, Cassity MB, Park J, Gillis R, Corbo J, *et al.*, Groblewski PA, Lecoq JA,
> Sedigh-Sarvestani M. **A single complex V2 as an alternative hypothesis for the
> organization of mouse higher visual cortex.** *bioRxiv* (2026).
> [doi:10.1101/2024.10.10.617533](https://doi.org/10.1101/2024.10.10.617533)

The paper proposes that the cortex surrounding mouse V1, usually divided into about
eight higher visual areas (HVAs), may instead be a single, complex second visual area
(V2). It combines retinotopic simulations with a new Allen Institute
[OpenScope](https://www.allenneuraldynamics.org/projects/openscope) dataset of
[electrophysiological](https://openscopedatafront.web.app/projects/cross_species_v2_ephys) and [two-photon](https://openscopedatafront.web.app/projects/cross_species_v2_ophys) recordings across V1. This repo is the
single place for the paper's public data links and analysis code.

## Contents

| Paper figure | What it shows | Code | Data |
| --- | --- | --- | --- |
| **Figure 2** | Retinotopic (elastic-net) simulations of a single complex V2 | [`Figure2/`](Figure2/) *(review draft)* | Simulated |
| **Figure 4** | The OpenScope dataset and example neurons | [`ephys_getting_started.ipynb`](ephys_getting_started.ipynb), [`02_ophys_getting_started.ipynb`](02_ophys_getting_started.ipynb) | [DANDI:001568](https://dandiarchive.org/dandiset/001568), [DANDI:001709](https://dandiarchive.org/dandiset/001709) |
| **Figure 5** | A within-V1 benchmark for functional differences among HVAs | [`Figure5/`](Figure5/) | Bundled tables; raw recordings from [DANDI:001568](https://dandiarchive.org/dandiset/001568) and the [Allen Visual Coding – Neuropixels](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html) dataset |

```
MouseV2/
├── README.md                        this page, plus full dataset documentation below
├── ephys_getting_started.ipynb      stream and explore one Neuropixels session
├── 02_ophys_getting_started.ipynb   stream and explore one two-photon session
├── Figure5/                         self-contained reproduction of Figure 5
│   ├── README.md                    build, full reconstruction, and raw re-extraction
│   ├── Figure5.py                   quick build from the bundled, hash-locked tables
│   ├── reproduce.py                 full reconstruction (5,000 bootstrap replicates)
│   └── ...
└── Figure2/                         retinotopy simulations and frozen Figure 2 composition
```

## Getting started

Each part of the repository has **its own Python environment**, so each can pin the
library versions its results were produced with. Create a separate virtual
environment for each part you want to run, from the repository root.

```bash
git clone https://github.com/SarvestaniLab/MouseV2.git
cd MouseV2
```

### Reproduce Figure 2

Figure 2 uses its own locked Python 3.8 environment and frozen model checkpoints:

```bash
cd Figure2
uv sync --locked --group notebooks
uv run --locked --group notebooks python fig2/compose_figure.py
cd ..
```

See [Figure 2 reproduction instructions](Figure2/REPRODUCING.md) for model training
and tests, and [known limitations](Figure2/KNOWN_LIMITATIONS.md) for outstanding
scientific and release reviews. This is a review draft, not a finalized release.

### Explore the OpenScope data (Figure 4)

The notebooks stream one session directly from DANDI. No download, account, or API
key is needed. Python 3.10 or later:

```bash
python -m venv .venv-notebooks
source .venv-notebooks/bin/activate        # Windows: .venv-notebooks\Scripts\activate
pip install dandi pynwb remfile h5py numpy pandas matplotlib jupyterlab
jupyter lab
```

### Reproduce Figure 5

Figure 5 uses Python 3.12 and the exact versions in
[`Figure5/requirements.txt`](Figure5/requirements.txt):

```bash
python3.12 -m venv .venv-figure5           # Windows: py -3.12 -m venv .venv-figure5
source .venv-figure5/bin/activate          # Windows: .venv-figure5\Scripts\activate
pip install -r Figure5/requirements.txt
python Figure5/Figure5.py
```

This rebuilds the figure from the bundled analysis tables and checks every numerical
output against its locked SHA-256 hash. For a typeset-identical figure, install the
Liberation Sans font first. [`Figure5/README.md`](Figure5/README.md) covers the font,
the full statistical reconstruction, and re-extraction from the raw NWB recordings.

## Data

| Dataset | Contents | Access |
| --- | --- | --- |
| OpenScope MouseV2 – Ephys | Neuropixels 1.0 recordings at four V1 locations; 8 mice | [DANDI:001568](https://dandiarchive.org/dandiset/001568) |
| OpenScope MouseV2 – Ophys | Two-photon mesoscope imaging at 12 V1 locations; 6 mice | [DANDI:001709](https://dandiarchive.org/dandiset/001709) |
| Allen Brain Observatory Visual Coding – Neuropixels | V1 and HVA recordings ([Siegle et al., 2021](https://doi.org/10.1038/s41586-020-03171-x)); used in Figure 5 | [AllenSDK documentation](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html) |

Both OpenScope dandisets are open and documented in detail
[below](#the-openscope-dataset).

## Citation

If you use this code or the OpenScope dataset, please cite:

```bibtex
@article{rowley2026mousev2,
  title   = {A single complex {V2} as an alternative hypothesis for the organization
             of mouse higher visual cortex},
  author  = {Rowley, Declan P. and Cassity, Mary Beth and Park, Jimin and others
             and Sedigh-Sarvestani, Madineh},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.1101/2024.10.10.617533}
}
```

---

## The OpenScope dataset

### Overview

This dataset was collected by the Allen Institute as part of [OpenScope](https://www.allenneuraldynamics.org/projects/openscope) to test the unified mouse V2 hypothesis presented in [(Rowley et al., 2026)](https://www.biorxiv.org/content/10.1101/2024.10.10.617533v4). By releasing the dataset, we hope to allow others to independently test this hypothesis and other hypotheses surrounding mouse visual cortex organization.


The dataset includes electrophysiological recordings (Ephys) using four Neuropixels probes within V1 and two-photon measurements (Ophys) from 12 V1 locations, together with stimuli for receptive-field mapping and characterization of orientation (Ori), spatial-frequency (SF), and temporal-frequency (TF) tuning.

Generic probe placement (left) and two-photon imaging planes locations (right) are shown below. For two-photon imaging, 4 planes were imaged at 2 depths in a single session (i.e., all orange planes in the first session, all blue planes in the second session, etc.). This means there are 8 imaging planes per file for the ophys dataset.
<table align="center">
  <tr>
  <td><img width="272" height="238" alt="Asset 4" src="https://github.com/user-attachments/assets/be770cf2-9a23-4ec5-b660-c30060828a06" />
  <td><img width="234" height="208" alt="Asset 2" src="https://github.com/user-attachments/assets/5f0ca0a2-18c5-4e6d-bff5-6149b319199e" />
 </tr>
</table>

Visual area boundaries for V1 were obtained using intrinsic signal imaging (ISI). For a more detailed description of ISI, please see page 3 of the Visual Coding Overview [here](https://community.brain-map.org/t/documentation-brain-observatory/3026).

### Stimuli

To assess the spatial and feature tuning of V1 neurons recorded at different retinotopic V1 locations, the following stimulus set was used: 
- The stimuli for the electrophysiological recordings includes
  * gabor patches (fixed TF and SF with varying Ori: 0°, 45°, 90°)
  * full-field drifting gratings (TF: 1.0, 2.0, 4.0, 8.0, 15.0 Hz; SF: 0.02, 0.04, 0.08, 0.16, 0.32 cpd; Ori: 0°, 45°, 90°, 135°)
  * full-field flashes.   

- The stimuli for the two-photon measurements includes
  * gabor patches (fixed TF and SF with varying Ori: 0°, 45°, 90°)
  * full-field drifting gratings (TF: 1.0, 2.0, 4.0, 8.0, 15.0 Hz; SF: 0.02, 0.04, 0.08, 0.16, 0.32 cpd; Ori: 0°, 45°, 90°, 135°).   

An example truncated ephys session is shown below.

https://github.com/user-attachments/assets/1c861f92-a653-4379-990b-01a02dc8050d

### Recording methods and data structure
#### Ephys
##### Data recording methods

**Animal preparation and behavioral apparatus** Five male and 3 female wild type mice (99 to 114 days old) were used. They were prepared with a SHIELD cranial implant for repeated probe access, then habituated to the behavioral apparatus. Mice were head-fixed on a treadmill on which they could run freely, and viewed visual stimuli presented on a monitor to the right eye. Sessions were passive: no reward was delivered and mice were not restricted. Running speed and pupil position were recorded throughout. 

The behavioral and imaging apparatus for the ephys dataset is shown below.

<p align="center">
  <img width="556" height="355" alt="figure3_ephys" src="https://github.com/user-attachments/assets/b09c7ade-fbc4-49ac-88e6-d15d9341a95e" />
</p>

**Imaging** Recordings were performed using Neuropixels 1.0 probes. Up to six probes were inserted to 3.1 mm and retracted by 100 µm. The 348 closet recording sites to the probe tip were sampled using Open Ephys. Each mouse was recorded once.

**Data pre-processing** Spikes were sorted with Kilosort 4. Data is distributed in the Neurodata Without Borders format, with each unit (neuron) containing the spike times and the probe it was recorded from. 

##### Data structure

For session 817335, 

| Neural data | Stimulus data | Behavioural data |
|---|---|---|
| **`nwb.units`**<br>2403 units, one row per neuron, across 4 probes — ProbeA · ProbeB · ProbeC · ProbeE.<br>*Key columns* `spike_times` · `firing_rate` · `device_name` (which probe)<br><br>**Electrodes**<br>**`nwb.electrodes`**<br>1920 rows, one per recording channel (480/probe)<br>`location` · `group_name` · `channel_name` · `gain_to_physical_unit` (~0.195 µV/bit) · `rel_x` · `rel_y`<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br> | **`nwb.intervals`**<br>Four stimulus tables, one per block.<br>One row per trial.<br>*Shared columns*<br>`start_time` · `stop_time` (seconds, session clock)<br>`stim_name` · `stim_type` · `stim_index`<br><br>**RF block**<br>**`['receptive_field_block_presentations']`**<br>4860 rows · 0.25 s per trial, back-to-back<br>*Position* `x_position` · `y_position`<br>*Grating* `orientation` · `spatial_frequency` · `temporal_frequency` · `contrast`<br><br>**Tuning block**<br>**`['drifting_gratings_field_block_presentations']`**<br>1500 rows · 1.00 s per trial, 1.25 s apart<br>*Grating* `orientation` · `spatial_frequency` · `temporal_frequency` · `contrast`<br>No position columns — full field<br><br>**Flash block**<br>**`['flash_field_block_presentations']`**<br>300 rows · 0.25 s per trial, 2.00 s apart<br>Grating columns present but unused — full-field flash, no position<br><br>**Spontaneous block**<br>**`['spontaneous_presentations']`**<br>2 rows · gray screen, no stimulus parameters | **Locomotion**<br>**`nwb.processing['running']`**<br>320,687 samples · ~60 Hz<br>TimeSeries — timestamps on `.timestamps`<br>`['running_speed']` · cm/s, negative = backwards<br>`['running_wheel_rotation']` · radians<br><br>**Eye tracking**<br>**`nwb.processing['eye_tracking']`**<br>324,466 rows · ~60 Hz · one row per camera frame<br>`['corneal_reflection']` · `['ellipse']` · `['pupil']`<br>DynamicTables — timestamps are a column, not `.timestamps`<br>*All three tables have these columns*<br>`data_x` · `data_y` · `area` · `area_raw` · `width` · `height` · `angle` · `timestamps`<br>`reference_frame = 'nose'`<br>⚠️ `-1` marks a failed fit, not a value. Nothing is NaN.<br>`['likely_blink_times']`<br>TimeSeries, bool — timestamps on `.timestamps`<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br> |

An example breakdown of trials and times for one of the ephys sessions is shown below. 

<img width="4601" height="753" alt="817335_ephys_plot" src="https://github.com/user-attachments/assets/21a1acaf-042d-4b6a-8de3-b0feb9ffdedf" />

#### Ophys
##### Data recording methods

 
**Animal preparation and behavioral apparatus** were as described in the [Allen Brain Observatory Visual Coding: Overview technical whitepaper](https://s3.amazonaws.com/webflow-prod-assets/689cfbd308fa7373b604d290/68ee796f6f627db2434e459c_Documentation_Brain_Observatory-Visual_Coding_Overview.pdf).  Six female transgenic mice (96 to 145 days old, Cux2-CreERT2/wt;Camk2a-tTA/wt;Ai93(TITL-GCaMP6f)/wt), expressing GCaMP6f in excitatory neurons of layers 2/3 and 4, were used. They were prepared with a cranial window and headbar, then habituated to the behavioral apparatus. Mice were head-fixed on a treadmill on which they could run freely, and viewed visual stimuli presented on a monitor to the right eye. Sessions were passive: no reward was delivered and mice were not restricted. Running speed and pupil position were recorded throughout.

The behavioral and imaging apparatus for the ophys dataset is shown below.

<p align="center">
  <img width="556" height="355" alt="figure3_2p" src="https://github.com/user-attachments/assets/26dc2e76-3b43-4260-ba78-c1b377be1e6d" />
</p>

**Imaging** was performed on two Thorlabs mesoscopes. Eight planes (4 locations × 2 depths) were imaged simultaneously: four at 130–170 µm and four at 226–270 µm below the pia, within VISp. Planes were 512 × 512 pixels at 0.78 µm/pixel (399 µm field of view), acquired at 10.63 Hz on MESO.2 (809092, 810268, 826616, 826619) or 9.48 Hz on MESO.1 (815059, 823093). Each mouse was recorded in three sessions of ~67 min, sampling 12 locations tiling V1.

**Data pre-processing** was performed in accordance with the [AIND multiplane-ophys pipeline](https://github.com/AllenNeuralDynamics/aind-multiplane-ophys-pipeline). Frames were first de-interleaved into their constituent planes, which were motion-corrected in Suite2p. Regions of interest were detected and fluorescence extracted using a combination of Cellpose and Suite2p, neuropil contamination was subtracted, ΔF/F traces were computed, and spiking events were inferred for each ROI with the OASIS deconvolution library. Processed data are distributed in Neurodata Without Borders format, with each imaging plane stored as a separate processing module containing raw, neuropil-corrected, ΔF/F and event traces, segmentation masks and summary images.

##### Data structure

**One NWB file = one session.**<br>**One clock per session.** Every array carries its own timestamps, in seconds. Streams start at different times and run at different rates, so align on timestamps, not by index.

| Neural data | Stimulus data | Behavioural data |
|---|---|---|
| **`nwb.processing['VISp_0'] … ['VISp_7']`**<br>Eight modules, one per imaging plane — identical structure.<br>N frames × N ROIs.<br>9.5 or 10.6 Hz, by microscope.<br><br>**Traces** — all the same shape, each with its own timestamps.**One row per frame · one column per ROI.**<br>`['raw_timeseries']['ROI_fluorescence_timeseries']` · a.u.<br>`['neuropil_fluorescence_timeseries']` · a.u.<br>`['neuropil_corrected_timeseries']` · a.u.<br>`['dff_timeseries']['dff_timeseries']` · %<br>`['event_timeseries']` · AP-related events<br><br>**Segmentation**<br> **`['image_segmentation']['roi_table']`**<br>N rows, one per ROI<br>`is_soma` · `soma_probability` · `is_dendrite` · `dendrite_probability` · `image_mask` (512 × 512, use `> 0` for the footprint)<br><br>**Whole plane images**<br>**`['images']`**<br>`['average_projection']` · `['max_projection']` · `['segmentation_mask_image']` — 512 × 512 each<br>`['segmentation_mask_image']` labels every ROI in one array |**`nwb.intervals`**<br>Two stimulus tables, one per block.<br>One row per trial.<br>*Shared columns*<br>`start_time` · `stop_time` (seconds, session clock)<br>`stim_name` · `stim_type` · `stim_index`<br><br>**RF block**<br>**`['receptive_field_block_presentations']`**<br>N rows · 0.25 s per trial, back-to-back<br>*Position* `x_position` · `y_position`<br>*Grating* `orientation` · `spatial_frequency` · `temporal_frequency` · `contrast`<br>**Tuning block**<br>**`['drifting_gratings_field_block_presentations']`**<br>N rows · 1.0 s per trial, 1.25 s apart<br>*Grating* `orientation` · `spatial_frequency` · `temporal_frequency` · `contrast`<br>No position columns — full field<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp; | **Locomotion** <br> **`nwb.processing['running']`**<br>N samples · ~60 Hz<br>TimeSeries — timestamps on `.timestamps`<br>`['running_speed']` · cm/s, negative = backwards<br>`['running_wheel_rotation']` · radians<br><br>**Eye tracking**<br>**`nwb.processing['eye_tracking']`**<br>N rows · ~60 Hz · one row per camera frame<br>`['pupil']` · `['corneal_reflection']` · `['ellipse']`<br>&emsp;**DynamicTables — timestamps are a column**, not `.timestamps`<br>*All three tables have these columns*<br>`data_x` · `data_y` · `width` · `height` · `angle`<br>`area` · `area_raw` · `timestamps`<br>`reference_frame = 'nose'`<br>⚠️ `−1` marks a failed fit, not a value. Nothing is NaN.<br>`['likely_blink_times']`<br>TimeSeries, bool — timestamps on `.timestamps`<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp;<br>&nbsp; |

An example breakdown of trials and times for one of the ophys sessions is shown below.

<img width="4601" height="753" alt="809092_ophys_plot" src="https://github.com/user-attachments/assets/9105fce1-ff95-44d3-af36-49092eea584c" />

### Data access with DANDI CLI


The project is split across two dandisets, one per recording modality, **001568** for Ephys and **001709** for Ophys. <br>
One NWB file per session in both. They share the same layout and the same access
code — only the identifier and the file names differ. You can either **download** a
file with the DANDI command-line tool or **stream** it, reading only the bytes you
actually touch. <br>

For the DANDI CLI tool itself, see the archive's guides on
[downloading](https://docs.dandiarchive.org/user-guide-using/accessing-data/downloading/)
and [streaming](https://docs.dandiarchive.org/user-guide-using/accessing-data/streaming/).

```bash
pip install dandi pynwb remfile h5py
```

#### File paths

Named from the subject and the session start time, with the modality in both the
session token and the suffix:

```
Ephys: sub-<subject>/sub-<subject>_ses-ecephys-<subject>-<YYYY-MM-DD>-<HH-MM-SS>_ecephys.nwb 
Ophys: sub-<subject>/sub-<subject>_ses-multiplane-ophys-<subject>-<YYYY-MM-DD>-<HH-MM-SS>_ophys.nwb
```

**Show the paths and sizes for one subject, in bash**
```bash
dandi ls "dandi://DANDI/001568/sub-810531/"
```

**Show the paths and sizes for a whole dandiset**
```python
from dandi.dandiapi import DandiAPIClient

DANDISET = "001568"          # ephys;  use "001709" for ophys

with DandiAPIClient() as client:
    for asset in client.get_dandiset(DANDISET).get_assets():
        print(f"{asset.size / 1e9:5.2f} GB  {asset.path}")
```

#### Stream one file

```python
import h5py, pynwb, remfile
from dandi.dandiapi import DandiAPIClient

DANDISET = "001568"
PATH = ("sub-810531/"
        "sub-810531_ses-ecephys-810531-2025-09-17-15-14-30_ecephys.nwb")

# for the ophys dandiset instead:
# DANDISET = "001709"
# PATH = ("sub-809092/"
#         "sub-809092_ses-multiplane-ophys-809092-2025-10-01-12-34-23_ophys.nwb")

# find the file (asset) and its URL
asset = DandiAPIClient().get_dandiset(DANDISET).get_asset_by_path(PATH)
url = asset.get_content_url(follow_redirects=1, strip_query=False)
# stream the file
handle = h5py.File(remfile.File(url), "r")
io = pynwb.NWBHDF5IO(file=handle, mode="r")
nwbfile = io.read()
```

`follow_redirects=1` resolves the archive's redirect to storage; `strip_query=False`
keeps the signature that makes the resulting URL readable. Call `io.close()` and
`handle.close()` when done.

#### Download one file
**Command line**
```bash
dandi download "dandi://DANDI/001568/sub-810531/sub-810531_ses-ecephys-810531-2025-09-17-15-14-30_ecephys.nwb"
```
**Script**
```python
nwbfile = pynwb.NWBHDF5IO(PATH, mode="r").read()
```
