# Cohort-scale Wilson-Cowan inference from widefield calcium imaging

Compact reproducibility repository for the manuscript:

**Regression-Assisted Gradient-Enhanced Physics-Informed Neural Networks for Cohort-Scale Wilson-Cowan Inference from Widefield Calcium Imaging**

This public release keeps the same design philosophy as the earlier single-session repository: one frozen configuration, one Python entry point, one Slurm entry point, focused source modules, lightweight derived results, and no historical `v1/v2/final` development scripts.

## Repository structure

```text
wilson-cowan-cohort-reproducibility/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── environment.yml
├── config.yaml
├── run_pipeline.py
├── submit_pipeline.sh
├── data/
│   ├── README.md
│   └── preprocessing_manifest_40.csv
├── src/
│   ├── model_selection.py
│   ├── gpinn_core.py
│   ├── reconstruction.py
│   ├── reconstruction_preflight.py
│   ├── reconstruction_aggregate.py
│   ├── dynamics.py
│   ├── longitudinal.py
│   ├── hierarchical.py
│   ├── reporting.py
│   └── supplementary_figure.py
├── slurm/                  # portable Alliance/Narval job templates
├── scripts/                # release collection + validation only
├── results/
│   ├── README.md
│   └── summary/            # lightweight manuscript-level derived outputs
└── docs/
    ├── SCIENTIFIC_GUARDRAILS.md
    └── RELEASE_CHECKLIST.md
```

## Data and preprocessing

Raw BraiDyn-BC data are not duplicated here. The derivative-aware preprocessing workflow is maintained separately, as in the earlier single-session release, rather than vendored into this repository.

Companion preprocessing repository:

**https://github.com/abrahamsegun001/derivative-aware-widefield**

This modelling repository expects the final 64 × 64, 30-Hz processed chunks under:

```text
data/processed_64_linear/
```

The committed `data/preprocessing_manifest_40.csv` defines the 40 recording tasks used by the cohort analysis.

## Cohort

- 10 mice: GC48, GC51, GC52, GC53, GC54, GC60, GC61, GC63, GC66, GC73
- 4 sessions: day0, day1, day7, day15
- 40 recordings
- 20 contiguous 30-s chunks per recording
- 800 chunks in the structural model-selection analysis
- 738 inference-valid chunks in the primary nonlinear-dynamics analysis
- all 800 chunks retained in a prespecified sensitivity analysis

The mouse is the biological replication unit; chunks are repeated observations nested within recording/session and mouse.

## Installation

```bash
conda env create -f environment.yml
conda activate wilson-cowan-cohort
python run_pipeline.py verify
```

Install the PyTorch build appropriate for the local CUDA stack when running GPU reconstruction.

## Compact command interface

Each stage is available through the single root entry point:

```bash
python run_pipeline.py model-selection --help
python run_pipeline.py dynamics --help
python run_pipeline.py preflight
python run_pipeline.py reconstruct --help
python run_pipeline.py aggregate-reconstruction --help
python run_pipeline.py longitudinal --help
python run_pipeline.py hierarchical --help
python run_pipeline.py report --help
python run_pipeline.py validate-release
```

### 1. Frozen regression-assisted model selection

One recording per task:

```bash
python run_pipeline.py model-selection --task-id 0
```

After all 40 tasks:

```bash
python run_pipeline.py model-selection --aggregate
```

Primary output:

```text
outputs/cohort_model_selection_40/cohort_model_selection_800.csv
```

### 2. Equilibrium and bifurcation analysis

```bash
python run_pipeline.py dynamics
```

The scalar autonomous reduction is interpreted through fixed points, bistability, and saddle-node geometry. It is not used as a model of classical excitatory-inhibitory Hopf oscillations.

### 3. Fixed-parameter neural-state reconstruction

```bash
python run_pipeline.py preflight
python run_pipeline.py reconstruct --task-id 0
python run_pipeline.py aggregate-reconstruction
```

Each recording uses one deterministic inference-valid representative chunk, independent cold-start training, and fixed physical coefficients loaded from the selector. Validation is random-pointwise spatiotemporal interpolation within the selected 30-s chunk, not forecasting.

### 4. Longitudinal and hierarchical analyses

```bash
python run_pipeline.py longitudinal
python run_pipeline.py hierarchical
```

Repeated-measures inference is performed at the mouse level. Chunk-level and mouse-session variation is retained for hierarchical variance analysis.

### 5. Manuscript synthesis

```bash
python run_pipeline.py report
```

This regenerates manuscript-level tables and figures from the completed model-selection, dynamics, longitudinal, hierarchical, and reconstruction outputs.

## HPC / Slurm

The repository exposes one submission interface while retaining readable stage-specific templates:

```bash
./submit_pipeline.sh model-selection --account=<cpu-allocation>
./submit_pipeline.sh reconstruct --account=<gpu-allocation>
```

No personal cluster path or allocation is hard-coded in the scientific source.

## Derived model-parameter data

Small manuscript-level summaries are committed under `results/summary/`. Before the public release associated with the manuscript, collect and validate the complete lightweight derived parameter tables from the analysis workspace:

```bash
bash scripts/collect_release_outputs.sh /path/to/gpinn_project
python run_pipeline.py validate-release
```

The release validator checks the 800-chunk model-selection table, the 738-chunk inference-valid primary-dynamics table, the all-800-chunk sensitivity dynamics table, longitudinal and hierarchical outputs, and the 40-recording reconstruction metrics.

## Scientific interpretation

`D_E` is an effective homogeneous isotropic Laplacian spatial-coupling coefficient in the reduced Wilson-Cowan formulation, not a directly calibrated biological tissue diffusion constant. Selection of the reduced `D_E = 0` model means that the Laplacian term lacks stable support under the stated data, preprocessing, scales, and frozen selection criteria; it does not establish absence of cortical spatial coupling.

Model-implied bistability is reported in normalized-fluorescence coordinates. Fitted `w_EE` and `h` are effective model coefficients rather than direct physiological measurements.

## Reproducibility status

The cohort analysis outputs included in this repository have been collected from the completed Narval analyses and validated using the repository release checks.

The repository contains the analysis code and derived tables supporting the manuscript, including:

- model selection across 800 chunks;
- primary nonlinear-dynamics analysis across 738 inference-valid chunks;
- sensitivity analysis across all 800 chunks;
- fixed-parameter neural-state reconstruction across 40 recordings;
- longitudinal and hierarchical analyses across 10 mice.

A permanent archival DOI will be added after the `v1.0.0` release is deposited in Zenodo.


## License and citation

Released under the BSD 3-Clause License. `CITATION.cff` is included for GitHub citation support.
